from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch_geometric.loader import DataLoader

from .config import get_cfg, AppConfig
from .utils import compute_metrics, init_logging, mkdirp, save_json, set_seed, stratified_kfold
from .causal_brain_network_construction import GraphFolderDataset
from .causal_brain_graph_transformer import BrainCGT


def _make_run_dir(results_dir: str, tag: str = "exp") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / f"{tag}_{ts}"
    mkdirp(str(run_dir))
    mkdirp(str(run_dir / "checkpoints"))
    mkdirp(str(run_dir / "metrics"))
    mkdirp(str(run_dir / "explanations"))
    return run_dir


def _build_model(cfg: AppConfig) -> BrainCGT:
    m = BrainCGT(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        attn_dim=cfg.model.attn_dim,
        dropout=cfg.model.dropout,
        gamma=cfg.model.gamma_causal_bias,
        use_posenc=cfg.model.use_posenc,
        posenc_dim=cfg.model.posenc_dim,
        num_classes=cfg.model.num_classes,
    )
    return m


def _batch_to_device(batch, device):
    return batch.to(device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob_list: List[np.ndarray] = []

    for batch in loader:
        batch = _batch_to_device(batch, device)
        out = model(batch)
        prob = out["prob"].detach().cpu().numpy()
        pred = prob.argmax(axis=1)
        y = batch.y.view(-1).detach().cpu().numpy().astype(int)

        y_true.extend(y.tolist())
        y_pred.extend(pred.tolist())
        y_prob_list.append(prob)

    y_prob = np.concatenate(y_prob_list, axis=0) if len(y_prob_list) else np.zeros((0, model.head.classifier[-1].out_features))
    if y_prob.shape[1] == 2:
        y_prob_auc = y_prob[:, 1]
    else:
        y_prob_auc = y_prob
    metrics = compute_metrics(np.array(y_true), y_prob_auc, np.array(y_pred))
    return {"metrics": metrics, "y_true": np.array(y_true), "y_pred": np.array(y_pred), "y_prob": y_prob}


def _combat_correct_activity(graphs: List, site_ids: List[str]) -> None:
    """
    In-place ComBat harmonization on per-subject ROI activity scalars data.a [N,1].
    This uses neuroHarmonize and treats site_id as the batch effect.
    """
    try:
        from neuroHarmonize import harmonizationLearn, harmonizationApply
    except Exception as e:
        raise RuntimeError("neuroHarmonize is required when preprocess.combat=true.") from e

    X = np.stack([g.a.detach().cpu().numpy().reshape(-1) for g in graphs], axis=0)  # [S,N]
    covars = pd.DataFrame({"SITE": site_ids})
    # Call robustly across minor API variations
    learn_sig = harmonizationLearn.__code__.co_varnames
    if "batch_col" in learn_sig:
        model, X_h = harmonizationLearn(X, covars, batch_col="SITE")
    else:
        model, X_h = harmonizationLearn(X, covars, "SITE")
    apply_sig = harmonizationApply.__code__.co_varnames
    if "model" in apply_sig and "data" in apply_sig:
        X_h2 = harmonizationApply(X, covars, model)
    else:
        X_h2 = harmonizationApply(X, covars, model)

    X_h = np.asarray(X_h2, dtype=np.float32)
    for i, g in enumerate(graphs):
        g.a = torch.from_numpy(X_h[i].reshape(-1, 1)).to(g.a.device)


def train_kfold(cfg: AppConfig, graphs_index_csv: str, run_tag: str = "braincgt") -> Dict:
    set_seed(cfg.train.seed)
    run_dir = _make_run_dir(cfg.paths.results_dir, tag=run_tag)
    init_logging(str(run_dir / "train.log"))
    save_json(cfg.model_dump(), str(run_dir / "config.json"))
    logger.info(f"Run directory: {run_dir}\n")

    dataset = GraphFolderDataset(graphs_index_csv)
    df = dataset.df.copy()
    subject_ids = df["subject_id"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    folds = stratified_kfold(subject_ids, labels, cfg.train.folds, cfg.train.seed)
    all_fold_metrics = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_id, fold in enumerate(folds):
        logger.info(f"==== Fold {fold_id}/{len(folds)-1} ====")
        train_idx = fold["train"]
        test_idx = fold["test"]

        # Build train/val split from train_idx
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # 10% stratified val split (deterministic)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg.train.seed)
        tr_sub_idx, va_sub_idx = next(sss.split(df_train["subject_id"], df_train["label"]))
        df_tr = df_train.iloc[tr_sub_idx].reset_index(drop=True)
        df_va = df_train.iloc[va_sub_idx].reset_index(drop=True)

        # Load graphs for each subset (so we can optionally ComBat-correct)
        def load_subset(dfi: pd.DataFrame) -> List:
            graphs = []
            for p in dfi["graph_path"].tolist():
                graphs.append(torch.load(p))
            # attach identifiers
            for i, g in enumerate(graphs):
                g.subject_id = str(dfi.iloc[i]["subject_id"])
                g.site_id = str(dfi.iloc[i]["site_id"])
                g.y = torch.tensor([int(dfi.iloc[i]["label"])], dtype=torch.long)
            return graphs

        tr_graphs = load_subset(df_tr)
        va_graphs = load_subset(df_va)
        te_graphs = load_subset(df_test)

        if cfg.preprocess.combat:
            logger.info("Applying ComBat harmonization on activity scalars (train-fit, apply to all subsets).")
            # Fit on train and apply using the same model by learning on concatenated for simplicity.
            # neuroHarmonize does not expose a stable public 'apply' with a fitted model across versions,
            # so we harmonize each subset with the model fitted on train covariates via harmonizationApply.
            try:
                from neuroHarmonize import harmonizationLearn, harmonizationApply
                X_tr = np.stack([g.a.detach().cpu().numpy().reshape(-1) for g in tr_graphs], axis=0)
                cov_tr = pd.DataFrame({"SITE": [g.site_id for g in tr_graphs]})
                learn_sig = harmonizationLearn.__code__.co_varnames
                if "batch_col" in learn_sig:
                    model_h, X_tr_h = harmonizationLearn(X_tr, cov_tr, batch_col="SITE")
                else:
                    model_h, X_tr_h = harmonizationLearn(X_tr, cov_tr, "SITE")

                def apply_subset(graphs: List):
                    X = np.stack([g.a.detach().cpu().numpy().reshape(-1) for g in graphs], axis=0)
                    cov = pd.DataFrame({"SITE": [g.site_id for g in graphs]})
                    X_h = harmonizationApply(X, cov, model_h)
                    X_h = np.asarray(X_h, dtype=np.float32)
                    for i, g in enumerate(graphs):
                        g.a = torch.from_numpy(X_h[i].reshape(-1, 1))

                # Update train with learned harmonized values
                X_tr_h = np.asarray(X_tr_h, dtype=np.float32)
                for i, g in enumerate(tr_graphs):
                    g.a = torch.from_numpy(X_tr_h[i].reshape(-1, 1))
                apply_subset(va_graphs)
                apply_subset(te_graphs)
            except Exception as e:
                raise RuntimeError(f"ComBat harmonization failed: {e}") from e

        train_loader = DataLoader(tr_graphs, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(va_graphs, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(te_graphs, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

        model = _build_model(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)

        best_metric = -1e9
        best_epoch = -1
        patience = 0

        ckpt_dir = run_dir / "checkpoints"
        fold_ckpt_best = ckpt_dir / f"fold{fold_id}_best.pt"
        fold_ckpt_last = ckpt_dir / f"fold{fold_id}_last.pt"

        for epoch in range(1, cfg.train.epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = _batch_to_device(batch, device)
                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                    out = model(batch)
                    logits = out["logits"]
                    y = batch.y.view(-1)
                    loss = F.cross_entropy(logits, y)

                scaler.scale(loss).backward()
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.detach().cpu())
                n_batches += 1

            # Validation
            val_out = evaluate(model, val_loader, device)
            val_metrics = val_out["metrics"]
            monitor = val_metrics[cfg.train.monitor_metric]

            logger.info(f"Fold {fold_id} | Epoch {epoch:03d} | loss={total_loss/max(n_batches,1):.4f} | "
                        f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}\n")

            if monitor > best_metric:
                best_metric = monitor
                best_epoch = epoch
                patience = 0
                torch.save({"model": model.state_dict(), "cfg": cfg.model_dump(), "fold": fold_id, "epoch": epoch}, fold_ckpt_best)
            else:
                patience += 1
                if patience >= cfg.train.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best {cfg.train.monitor_metric}={best_metric:.4f}).\n")
                    break

            if epoch % cfg.train.checkpoint_every == 0:
                torch.save({"model": model.state_dict(), "cfg": cfg.model_dump(), "fold": fold_id, "epoch": epoch}, fold_ckpt_last)

        # Test evaluation with best checkpoint
        ckpt = torch.load(fold_ckpt_best, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_out = evaluate(model, test_loader, device)
        test_metrics = test_out["metrics"]

        fold_metrics = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "best_monitor": best_metric,
            "test": test_metrics,
        }
        all_fold_metrics.append(fold_metrics)
        save_json(fold_metrics, str(run_dir / "metrics" / f"fold{fold_id}_metrics.json"))
        logger.info(f"Fold {fold_id} test metrics: {test_metrics}\n")

    save_json({"folds": all_fold_metrics}, str(run_dir / "metrics" / "kfold_summary.json"))
    return {"run_dir": str(run_dir), "folds": all_fold_metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--graphs_dir", required=True, type=str, help="Directory containing graphs_index.csv and .pt graphs.")
    ap.add_argument("--override", action="append", default=[], help="Override config keys, e.g., model.hidden_dim=256")
    ap.add_argument("--run_tag", type=str, default="braincgt")
    args = ap.parse_args()

    cfg = get_cfg(args.config, overrides=args.override)
    graphs_index_csv = str(Path(args.graphs_dir) / "graphs_index.csv")
    train_kfold(cfg, graphs_index_csv, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
