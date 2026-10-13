"""Simulateur de contrats WTI, utilisé uniquement par les tests (aucune donnée réelle nécessaire)."""

import numpy as np
import pandas as pd

from svdcurve.data import contract_code, last_trade_date


def simulate_contracts(
    start: str = "2016-01-04",
    end: str = "2025-12-31",
    seed: int = 42,
    kappa: float = 1.3,
    sigma_chi: float = 0.33,
    sigma_xi: float = 0.14,
    rho: float = 0.25,
    mu_xi: float = 0.02,
    risk_premium: float = 0.03,
    kappa_fast: float = 9.0,
    sigma_fast: float = 0.45,
    obs_noise: float = 0.0015,
    s0: float = 65.0,
    life_months: int = 16,
) -> pd.DataFrame:
    """Simule un panneau de contrats à terme (**données fictives**).

    Modèle de type Schwartz–Smith à trois facteurs : ``ln S = chi + zeta + xi``
    avec un facteur de moyen terme ``chi`` (Ornstein–Uhlenbeck, retour à la
    moyenne lent), un facteur de très court terme ``zeta`` (Ornstein–Uhlenbeck
    rapide, qui n'affecte que les premières échéances) et un facteur de long
    terme ``xi`` (mouvement brownien avec dérive). Le prix à terme de maturité
    ``tau`` vaut ``ln F = exp(-kappa tau) chi + exp(-kappa_fast tau) zeta + xi
    + A(tau)``. Un petit bruit d'observation gaussien est ajouté, comme sur des
    données réelles.

    Ce mode sert uniquement à tester le code hors ligne : ses résultats ne
    disent rien sur le marché réel.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n, dt = len(dates), 1 / 252

    def ou_noise_scale(speed: float, sigma: float) -> tuple[float, float]:
        decay = np.exp(-speed * dt)
        return decay, sigma * np.sqrt((1 - decay**2) / (2 * speed))

    decay, sd_chi = ou_noise_scale(kappa, sigma_chi)
    decay_fast, sd_fast = ou_noise_scale(kappa_fast, sigma_fast)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    z3 = rng.standard_normal(n)
    chi, zeta, xi = np.zeros(n), np.zeros(n), np.zeros(n)
    xi[0] = np.log(s0)
    for t in range(1, n):
        chi[t] = decay * chi[t - 1] + sd_chi * z1[t]
        zeta[t] = decay_fast * zeta[t - 1] + sd_fast * z3[t]
        xi[t] = xi[t - 1] + (mu_xi - 0.5 * sigma_xi**2) * dt + sigma_xi * np.sqrt(dt) * z2[t]

    def log_forward(tau: np.ndarray, chi_t: np.ndarray, zeta_t: np.ndarray, xi_t: np.ndarray) -> np.ndarray:
        var = (
            (1 - np.exp(-2 * kappa * tau)) * sigma_chi**2 / (2 * kappa)
            + (1 - np.exp(-2 * kappa_fast * tau)) * sigma_fast**2 / (2 * kappa_fast)
            + sigma_xi**2 * tau
            + 2 * (1 - np.exp(-kappa * tau)) * rho * sigma_chi * sigma_xi / kappa
        )
        drift = (mu_xi - 0.5 * sigma_xi**2) * tau - (1 - np.exp(-kappa * tau)) * risk_premium / kappa
        return np.exp(-kappa * tau) * chi_t + np.exp(-kappa_fast * tau) * zeta_t + xi_t + drift + 0.5 * var

    start_ts, end_ts = dates[0], dates[-1]
    rows: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start_ts.year, start_ts.month, 1)
    last_delivery = end_ts + pd.DateOffset(months=life_months + 1)
    while cursor <= last_delivery:
        expiry = last_trade_date(cursor.year, cursor.month)
        mask = (dates > expiry - pd.DateOffset(months=life_months)) & (dates <= expiry)
        if mask.any():
            tau = (expiry - dates[mask]).days.to_numpy() / 365.25
            ln_f = log_forward(tau, chi[mask], zeta[mask], xi[mask]) + obs_noise * rng.standard_normal(mask.sum())
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates[mask],
                        "contract": contract_code(cursor.year, cursor.month),
                        "expiry": expiry,
                        "price": np.exp(ln_f),
                    }
                )
            )
        cursor += pd.DateOffset(months=1)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "expiry"]).reset_index(drop=True)
