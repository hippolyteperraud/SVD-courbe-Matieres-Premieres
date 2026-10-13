"""Tests de la PCA par SVD : on vérifie les propriétés démontrées dans le cours."""

import numpy as np
import pytest

from svdcurve.pca import SVDPCA, lambda_star, marchenko_pastur_median, omega, optimal_hard_threshold


@pytest.fixture
def data() -> np.ndarray:
    rng = np.random.default_rng(0)
    mixing = rng.standard_normal((6, 6))
    return rng.standard_normal((500, 6)) @ mixing + rng.standard_normal(6)


def test_variances_are_eigenvalues_of_covariance(data):
    pca = SVDPCA().fit(data)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(data, rowvar=False)))[::-1]
    np.testing.assert_allclose(pca.explained_variance_, eigenvalues, rtol=1e-10)


def test_components_are_eigenvectors_of_covariance(data):
    pca = SVDPCA().fit(data)
    cov = np.cov(data, rowvar=False)
    for variance, v in zip(pca.explained_variance_, pca.components_):
        np.testing.assert_allclose(cov @ v, variance * v, atol=1e-9)


def test_components_orthonormal_and_scores_uncorrelated(data):
    pca = SVDPCA().fit(data)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(6), atol=1e-12)
    score_cov = np.cov(pca.scores_, rowvar=False)
    np.testing.assert_allclose(score_cov, np.diag(pca.explained_variance_), atol=1e-9)


def test_full_reconstruction_is_exact(data):
    pca = SVDPCA().fit(data)
    np.testing.assert_allclose(pca.reconstruct(6), data, atol=1e-10)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_eckart_young_error_matches_discarded_singular_values(data, k):
    pca = SVDPCA().fit(data)
    assert pca.frobenius_error(k) == pytest.approx(pca.theoretical_error(k), rel=1e-10, abs=1e-10)


def test_truncated_svd_beats_random_rank_k_projections(data):
    """Eckart–Young : aucune autre approximation de rang k n'est meilleure."""
    pca = SVDPCA().fit(data)
    centered = data - data.mean(axis=0)
    rng = np.random.default_rng(1)
    k = 2
    for _ in range(50):
        q, _ = np.linalg.qr(rng.standard_normal((6, k)))
        error = np.linalg.norm(centered - centered @ q @ q.T, "fro")
        assert error >= pca.theoretical_error(k) - 1e-9


def test_transform_matches_scores(data):
    pca = SVDPCA().fit(data)
    np.testing.assert_allclose(pca.transform(data), pca.scores_, atol=1e-10)


def test_sign_convention_is_deterministic_and_applied():
    rng = np.random.default_rng(3)
    grid = np.linspace(0, 1, 12)
    # trois facteurs : niveau, pente, courbure
    factors = np.vstack([np.ones(12), grid - 0.5, 1 - 4 * (grid - 0.5) ** 2])
    sample = rng.standard_normal((800, 3)) * [1.0, 0.5, 0.2] @ factors
    first = SVDPCA().fit(sample)
    second = SVDPCA().fit(-sample)  # signes opposés en entrée, mêmes conventions en sortie
    np.testing.assert_allclose(first.components_[:3], second.components_[:3], atol=1e-8)
    assert first.components_[0].sum() > 0
    assert first.components_[1][-1] > first.components_[1][0]
    assert first.components_[2][6] > 0.5 * (first.components_[2][0] + first.components_[2][-1])


def test_energy_rule(data):
    pca = SVDPCA().fit(data)
    k = pca.n_components_for_energy(0.9)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    assert cumulative[k - 1] >= 0.9
    assert k == 1 or cumulative[k - 2] < 0.9


def test_invalid_inputs():
    with pytest.raises(ValueError):
        SVDPCA().fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        SVDPCA().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


# ------------------------------------------------------------ Gavish & Donoho
def test_lambda_star_square_case():
    assert lambda_star(1.0) == pytest.approx(4 / np.sqrt(3), rel=1e-12)


def test_omega_square_case_is_2_858():
    assert omega(1.0) == pytest.approx(2.858, abs=2e-3)


@pytest.mark.parametrize("beta", [0.05, 0.25, 0.5, 0.75, 1.0])
def test_omega_matches_published_polynomial_approximation(beta):
    approx = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    assert omega(beta) == pytest.approx(approx, rel=0.02)


def test_marchenko_pastur_median_is_between_support_bounds():
    for beta in (0.1, 0.5, 1.0):
        median = marchenko_pastur_median(beta)
        assert (1 - np.sqrt(beta)) ** 2 < median < (1 + np.sqrt(beta)) ** 2


def test_hard_threshold_recovers_low_rank_signal():
    """Signal de rang 2 noyé dans un bruit gaussien (cf. exemple du chapitre)."""
    rng = np.random.default_rng(5)
    n = 400
    u, _ = np.linalg.qr(rng.standard_normal((n, 2)))
    v, _ = np.linalg.qr(rng.standard_normal((n, 2)))
    signal = u @ np.diag([60.0, 40.0]) @ v.T
    noisy = signal + rng.standard_normal((n, n))
    s = np.linalg.svd(noisy, compute_uv=False)
    known = optimal_hard_threshold(s, n, n, sigma=1.0)
    unknown = optimal_hard_threshold(s, n, n)
    assert known.rank == 2 and unknown.rank == 2
    assert known.tau == pytest.approx(4 / np.sqrt(3) * np.sqrt(n), rel=1e-9)
