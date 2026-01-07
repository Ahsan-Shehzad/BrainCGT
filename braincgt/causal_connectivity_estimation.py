from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import fdrcorrection

from .config import AppConfig
from .utils import standardize_timeseries, save_json, mkdirp


@dataclass
class GrangerCausalityEstimator:
    cfg: AppConfig

    def _fit_var(self, Y: np.ndarray, cols: list[str]) -> Tuple[object, int]:
        df = pd.DataFrame(Y.T, columns=cols)  # [T,d]
        model = VAR(df)
        if self.cfg.granger.bic_select:
            res = model.fit(maxlags=self.cfg.granger.max_lag_p, ic="bic")
        else:
            res = model.fit(self.cfg.granger.max_lag_p)
        return res, int(res.k_ar)

    def fit_transform(self, X: np.ndarray, module_id: np.ndarray) -> Dict:
        """
        Estimate directed adjacency A [N,N] using module-wise multivariate Granger causality.

        - For intra-module edges (m==n): fit VAR on module signals; test all ordered pairs within module.
        - For inter-module edges (m!=n): fit VAR on union of (m,n); test only cross-module ordered pairs.
        - Apply global FDR across all tested directed pairs.
        - Scale retained F-statistics to [0,1] by global min-max.
        """
        X = np.asarray(X, dtype=np.float32)
        module_id = np.asarray(module_id, dtype=np.int64)
        N = X.shape[0]
        if self.cfg.granger.standardize:
            X = standardize_timeseries(X)

        M = int(module_id.max() + 1)
        pvals = np.ones((N, N), dtype=np.float64)
        fstats = np.zeros((N, N), dtype=np.float64)
        tested_mask = np.zeros((N, N), dtype=bool)

        p_selected: Dict[str, int] = {}

        # Intra-module
        for m in range(M):
            idx = np.where(module_id == m)[0]
            if len(idx) < 3:
                continue
            Y = X[idx, :]
            cols = [f"roi_{i}" for i in idx.tolist()]
            try:
                res, p = self._fit_var(Y, cols)
                p_selected[f"intra_{m}"] = p
            except Exception as e:
                logger.warning(f"VAR fit failed for intra module {m}: {e}")
                continue

            for ii, i in enumerate(idx):
                for jj, j in enumerate(idx):
                    if i == j:
                        continue
                    try:
                        test = res.test_causality(caused=[f"roi_{j}"], causing=[f"roi_{i}"], kind="f")
                        pvals[i, j] = float(test.pvalue)
                        fstats[i, j] = float(test.test_statistic)
                        tested_mask[i, j] = True
                    except Exception:
                        continue

        # Inter-module (unordered pairs, test directed cross edges)
        for m in range(M):
            for n in range(m + 1, M):
                idx_m = np.where(module_id == m)[0]
                idx_n = np.where(module_id == n)[0]
                if len(idx_m) < 2 or len(idx_n) < 2:
                    continue
                idx = np.concatenate([idx_m, idx_n])
                Y = X[idx, :]
                cols = [f"roi_{i}" for i in idx.tolist()]
                key = f"inter_{m}_{n}"
                try:
                    res, p = self._fit_var(Y, cols)
                    p_selected[key] = p
                except Exception as e:
                    logger.warning(f"VAR fit failed for inter modules {m},{n}: {e}")
                    continue

                # m -> n and n -> m
                for i in idx_m:
                    for j in idx_n:
                        try:
                            test = res.test_causality(caused=[f"roi_{j}"], causing=[f"roi_{i}"], kind="f")
                            pvals[i, j] = float(test.pvalue)
                            fstats[i, j] = float(test.test_statistic)
                            tested_mask[i, j] = True
                        except Exception:
                            pass
                        try:
                            test = res.test_causality(caused=[f"roi_{i}"], causing=[f"roi_{j}"], kind="f")
                            pvals[j, i] = float(test.pvalue)
                            fstats[j, i] = float(test.test_statistic)
                            tested_mask[j, i] = True
                        except Exception:
                            pass

        tested_p = pvals[tested_mask]
        if tested_p.size == 0:
            A = np.zeros((N, N), dtype=np.float32)
            return {"A": A, "p_selected": p_selected, "pvals": pvals.astype(np.float32), "fstats": fstats.astype(np.float32)}

        reject, pvals_corr = fdrcorrection(tested_p, alpha=self.cfg.granger.fdr_alpha)
        # Build A from significant edges
        A = np.zeros((N, N), dtype=np.float32)
        sig_idx = np.where(tested_mask)
        # Map reject array back to matrix indices
        flat_indices = list(zip(sig_idx[0].tolist(), sig_idx[1].tolist()))
        f_sig = []
        for k, (i, j) in enumerate(flat_indices):
            if reject[k]:
                f_sig.append(fstats[i, j])
        if len(f_sig) == 0:
            return {"A": A, "p_selected": p_selected, "pvals": pvals.astype(np.float32), "fstats": fstats.astype(np.float32)}

        f_min = float(np.min(f_sig))
        f_max = float(np.max(f_sig))
        denom = (f_max - f_min) if (f_max > f_min) else 1.0

        for k, (i, j) in enumerate(flat_indices):
            if reject[k]:
                A[i, j] = float((fstats[i, j] - f_min) / denom)

        return {"A": A.astype(np.float32), "p_selected": p_selected, "pvals": pvals.astype(np.float32), "fstats": fstats.astype(np.float32)}
