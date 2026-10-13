"""PCA par SVD et choix du nombre de composantes.

Ce module ne dépend que de NumPy et SciPy : la PCA est écrite « à la main » à
partir de la SVD (chapitre 1 de *Data-Driven Science and Engineering*, Brunton
et Kutz) afin de rendre explicite le lien entre les objets de la SVD et ceux de
la PCA :

=========================  ==========================================
SVD de B = X - moyenne     Interprétation en PCA
=========================  ==========================================
``Vt`` (lignes)            composantes principales (« loadings »)
``U * s``                  scores (coordonnées des observations)
``s**2 / (n - 1)``         variances expliquées (valeurs propres de C)
=========================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize


class SVDPCA:
    """Analyse en composantes principales calculée par SVD économique.

    Pour une matrice de données ``X`` de taille ``(n_obs, n_features)`` (une
    observation par ligne), on forme ``B = X - moyenne`` puis on calcule la SVD
    économique ``B = U @ diag(s) @ Vt``. La matrice de covariance vaut alors
    ``C = B.T @ B / (n_obs - 1) = V @ diag(s**2 / (n_obs - 1)) @ V.T`` : les
    colonnes de ``V`` sont ses vecteurs propres, et ``s**2 / (n_obs - 1)`` ses
    valeurs propres.

    Parameters
    ----------
    fix_signs
        Les vecteurs singuliers ne sont définis qu'au signe près. Si ``True``
        (défaut), on impose une convention déterministe, utile pour comparer
        deux exécutions :

        - composante 1 : somme des coefficients positive (« niveau ») ;
        - composante 2 : dernier coefficient supérieur au premier (« pente ») ;
        - composante 3 : milieu au-dessus de la moyenne des extrémités
          (« courbure », en forme de bosse) ;
        - composantes suivantes : plus grand coefficient en valeur absolue
          positif.
    """

    def __init__(self, fix_signs: bool = True) -> None:
        self.fix_signs = fix_signs

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> "SVDPCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X doit être une matrice (n_obs, n_features).")
        n_obs, n_features = X.shape
        if n_obs < 2:
            raise ValueError("Il faut au moins deux observations.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contient des valeurs non finies (NaN ou inf).")

        self.n_obs_ = n_obs
        self.n_features_ = n_features
        self.mean_ = X.mean(axis=0)
        B = X - self.mean_

        # SVD économique : U (n_obs x r), s (r,), Vt (r x n_features), r = min(n, p)
        U, s, Vt = np.linalg.svd(B, full_matrices=False)
        if self.fix_signs:
            U, Vt = _apply_sign_convention(U, Vt)

        self.singular_values_ = s
        self.components_ = Vt
        self.scores_ = U * s
        self.explained_variance_ = s**2 / (n_obs - 1)
        self.explained_variance_ratio_ = s**2 / np.sum(s**2)
        self._B = B
        return self

    # ------------------------------------------------------------ utilities
    def transform(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        """Coordonnées de ``X`` dans la base des ``k`` premières composantes."""
        k = self._check_k(k)
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_[:k].T

    def reconstruct(self, k: int) -> np.ndarray:
        """Approximation de rang ``k`` des données (théorème d'Eckart–Young)."""
        k = self._check_k(k)
        return self.scores_[:, :k] @ self.components_[:k] + self.mean_

    def reconstruct_from_scores(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Reconstruit des observations à partir de leurs ``k`` premiers scores."""
        k = self._check_k(k)
        return np.asarray(scores)[..., :k] @ self.components_[:k] + self.mean_

    def frobenius_error(self, k: int) -> float:
        """Erreur de Frobenius ``||B - B_k||_F`` du rang ``k`` (calcul direct)."""
        k = self._check_k(k)
        B_k = self.scores_[:, :k] @ self.components_[:k]
        return float(np.linalg.norm(self._B - B_k, "fro"))

    def theoretical_error(self, k: int) -> float:
        """Même erreur, lue sur le spectre : ``sqrt(sum_{j>k} sigma_j^2)``.

        C'est exactement le contenu quantitatif d'Eckart–Young : l'erreur de la
        meilleure approximation de rang ``k`` ne dépend que des valeurs
        singulières écartées.
        """
        k = self._check_k(k)
        return float(np.sqrt(np.sum(self.singular_values_[k:] ** 2)))

    def n_components_for_energy(self, threshold: float) -> int:
        """Plus petit ``k`` dont la part de variance cumulée atteint ``threshold``."""
        cumulative = np.cumsum(self.explained_variance_ratio_)
        return int(np.searchsorted(cumulative, threshold - 1e-12) + 1)

    def _check_k(self, k: int | None) -> int:
        r = len(self.singular_values_)
        if k is None:
            return r
        if not 0 <= k <= r:
            raise ValueError(f"k doit être compris entre 0 et {r}.")
        return int(k)


def _apply_sign_convention(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixe le signe de chaque couple (colonne de U, ligne de Vt)."""
    U, Vt = U.copy(), Vt.copy()
    p = Vt.shape[1]
    for i in range(Vt.shape[0]):
        v = Vt[i]
        if i == 0:
            key = v.sum()
        elif i == 1:
            key = v[-1] - v[0]
        elif i == 2:
            key = v[p // 2] - 0.5 * (v[0] + v[-1])
        else:
            key = v[np.argmax(np.abs(v))]
        if key < 0:
            Vt[i] *= -1
            U[:, i] *= -1
    return U, Vt


# --------------------------------------------------------------------------
# Seuil dur optimal de Gavish et Donoho (section « Troncature » du chapitre 1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdResult:
    tau: float
    rank: int
    beta: float
    method: str


def lambda_star(beta: float) -> float:
    r"""Coefficient :math:`\lambda(\beta)` du seuil optimal à bruit connu.

    Vaut :math:`4/\sqrt{3}` pour une matrice carrée (``beta = 1``).
    """
    return float(np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1))))


def marchenko_pastur_median(beta: float) -> float:
    r"""Médiane :math:`\mu_\beta` de la loi de Marchenko–Pastur de rapport ``beta``.

    Solution de :math:`\int_{\beta_-}^{\mu_\beta} f_\beta(t)\,dt = 1/2` avec la
    densité :math:`f_\beta(t)=\sqrt{(\beta_+-t)(t-\beta_-)}/(2\pi\beta t)` et
    :math:`\beta_\pm=(1\pm\sqrt\beta)^2`.
    """
    if not 0 < beta <= 1:
        raise ValueError("beta doit appartenir à ]0, 1].")
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2

    def density(t: float) -> float:
        return np.sqrt(max((hi - t) * (t - lo), 0.0)) / (2 * np.pi * beta * t)

    def cdf_minus_half(m: float) -> float:
        return integrate.quad(density, lo, m, limit=200)[0] - 0.5

    return float(optimize.brentq(cdf_minus_half, lo + 1e-12, hi - 1e-12, xtol=1e-12))


def omega(beta: float) -> float:
    r"""Coefficient :math:`\omega(\beta)=\lambda(\beta)/\sqrt{\mu_\beta}` (bruit inconnu).

    Vaut environ 2,858 pour ``beta = 1``.
    """
    return lambda_star(beta) / float(np.sqrt(marchenko_pastur_median(beta)))


def optimal_hard_threshold(
    singular_values: np.ndarray,
    n_rows: int,
    n_cols: int,
    sigma: float | None = None,
) -> ThresholdResult:
    """Seuil dur optimal pour la troncature d'une SVD (Gavish & Donoho, 2014).

    Modèle : ``X = X_vrai + sigma * Bruit`` avec ``X_vrai`` de rang faible et
    ``Bruit`` à coefficients i.i.d. gaussiens centrés réduits.

    - ``sigma`` connu : ``tau = lambda(beta) * sqrt(max(n, p)) * sigma`` ;
    - ``sigma`` inconnu (cas courant) : ``tau = omega(beta) * mediane(s)``.
    """
    s = np.asarray(singular_values, dtype=float)
    beta = min(n_rows, n_cols) / max(n_rows, n_cols)
    if sigma is not None:
        tau = lambda_star(beta) * np.sqrt(max(n_rows, n_cols)) * sigma
        method = "bruit connu"
    else:
        tau = omega(beta) * float(np.median(s))
        method = "bruit inconnu"
    return ThresholdResult(tau=float(tau), rank=int(np.sum(s > tau)), beta=float(beta), method=method)
