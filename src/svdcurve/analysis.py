"""Analyse en composantes principales (par SVD) des rendements de la courbe à terme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CurvePanel
from .pca import SVDPCA, optimal_hard_threshold

TRADING_DAYS = 252
FACTOR_NAMES = ("Niveau", "Pente", "Courbure")


@dataclass
class AnalysisResult:
    panel: CurvePanel
    pca: SVDPCA
    metrics: dict
    scores: pd.DataFrame  # scores journaliers des trois premiers facteurs


def run_analysis(panel: CurvePanel, n_factors: int = 3) -> AnalysisResult:
    """Applique la SVD aux rendements centrés et calcule tous les indicateurs du rapport."""
    returns = panel.returns
    X = returns.to_numpy()
    n_obs, n_mat = X.shape

    pca = SVDPCA().fit(X)
    s = pca.singular_values_

    # --- 1. Lien SVD <-> covariance (équation (3) du chapitre) ------------------
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    eig_gap = float(np.max(np.abs(pca.explained_variance_ - eigenvalues) / eigenvalues))

    # --- 2. Variance expliquée et nombre de facteurs ----------------------------
    ratio = pca.explained_variance_ratio_
    threshold = optimal_hard_threshold(s, n_obs, n_mat)

    # --- 3. Les trois facteurs : niveau, pente, courbure ------------------------
    loadings = pca.components_[:n_factors]
    # Indicateurs « faits main » de la forme de la courbe, sur des blocs de maturités :
    # court = premier tiers, long = dernier tiers, milieu = le reste.
    third = n_mat // 3
    short, middle, long_ = returns.iloc[:, :third], returns.iloc[:, third : n_mat - third], returns.iloc[:, -third:]
    proxies = pd.DataFrame(
        {
            "Niveau": returns.mean(axis=1),
            "Pente": long_.mean(axis=1) - short.mean(axis=1),
            "Courbure": middle.mean(axis=1) - 0.5 * (short.mean(axis=1) + long_.mean(axis=1)),
        }
    )
    scores = pd.DataFrame(pca.scores_[:, :n_factors], index=returns.index, columns=FACTOR_NAMES[:n_factors])
    correlations = [float(np.corrcoef(scores[name], proxies[name])[0, 1]) for name in scores.columns]
    sign_changes = [int(np.sum(np.diff(np.sign(v[np.abs(v) > 1e-3 * np.abs(v).max()])) != 0)) for v in loadings]

    # --- 4. Reconstruction : théorème d'Eckart–Young -----------------------------
    ks = list(range(n_mat + 1))
    errors = [pca.frobenius_error(k) for k in ks]
    theory = [pca.theoretical_error(k) for k in ks]
    rmse_bp = [e / np.sqrt(n_obs * n_mat) * 1e4 for e in errors]
    relative_error = [e / errors[0] for e in errors]

    day_norm = np.linalg.norm(X - pca.mean_, axis=1)
    day = int(np.argmax(day_norm))
    example = {
        "date": str(returns.index[day].date()),
        "observed": X[day].tolist(),
        **{f"k{k}": pca.reconstruct_from_scores(pca.scores_[day], k).tolist() for k in (1, 2, 3)},
    }

    # --- 5. Forme de la courbe observée -----------------------------------------
    price_slope = np.log(panel.prices.iloc[:, -1] / panel.prices.iloc[:, 0])
    i_back, i_cont = price_slope.idxmin(), price_slope.idxmax()
    median_date = price_slope.index[(price_slope - price_slope.median()).abs().argmin()]
    shape = {
        "share_backwardation": float((price_slope < 0).mean()),
        "dates": {"backwardation": str(i_back.date()), "median": str(median_date.date()), "contango": str(i_cont.date())},
        "slope_pct": {
            "backwardation": float(100 * price_slope[i_back]),
            "median": float(100 * price_slope[median_date]),
            "contango": float(100 * price_slope[i_cont]),
        },
    }

    metrics = {
        "n_obs": int(n_obs),
        "n_maturities": int(n_mat),
        "data": panel.info,
        "vol_annual_pct": (100 * np.sqrt(TRADING_DAYS) * X.std(axis=0, ddof=1)).tolist(),
        "singular_values": s.tolist(),
        "variance_ratio": ratio.tolist(),
        "variance_cumulative": np.cumsum(ratio).tolist(),
        "k_energy_90": pca.n_components_for_energy(0.90),
        "k_energy_99": pca.n_components_for_energy(0.99),
        "gavish_donoho": {
            "tau": threshold.tau,
            "rank": threshold.rank,
            "beta": threshold.beta,
            "median_singular_value": float(np.median(s)),
        },
        "eigendecomposition_max_relative_gap": eig_gap,
        "loadings": loadings.tolist(),
        "loading_sign_changes": sign_changes,
        "factor_names": list(scores.columns),
        "proxy_correlations": correlations,
        "reconstruction": {
            "k": ks,
            "rmse_bp_per_day": rmse_bp,
            "relative_error": relative_error,
            "max_gap_to_theory": float(np.max(np.abs(np.array(errors) - np.array(theory)))),
        },
        "example_day": example,
        "curve_shape": shape,
        "factor_vol_annual_pct": (100 * np.sqrt(TRADING_DAYS) * scores.std(ddof=1)).tolist(),
    }
    return AnalysisResult(panel=panel, pca=pca, metrics=metrics, scores=scores)
