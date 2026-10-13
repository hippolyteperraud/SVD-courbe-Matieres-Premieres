"""Données : contrats à terme WTI (Yahoo Finance).

Format commun (« panneau de contrats »), une ligne par contrat et par jour :

==========  =====================================================
``date``    jour de cotation
``contract``  code du contrat, par exemple ``CLZ26`` (décembre 2026)
``expiry``  dernier jour de cotation du contrat
``price``   prix de règlement (``Close``), en USD par baril
==========  =====================================================

La fonction :func:`build_curve_panel` transforme ce panneau en matrice de
rendements « par rang de maturité » (1er contrat, 2e contrat, ...), calculés
**contrat par contrat** : le rendement du rang *k* le jour *t* est celui du
contrat qui occupait le rang *k* la veille. On évite ainsi les faux sauts à la
date de roulement (roll) des séries « continues » classiques.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

MONTH_CODES = "FGHJKMNQUVXZ"  # janvier ... décembre (convention CME)
REQUIRED_COLUMNS = ("date", "contract", "expiry", "price")


# --------------------------------------------------------------------------
# Calendrier du contrat WTI
# --------------------------------------------------------------------------
def last_trade_date(year: int, month: int) -> pd.Timestamp:
    """Dernier jour de cotation du contrat WTI livrable au mois ``(year, month)``.

    Règle CME : la cotation s'arrête 3 jours ouvrés avant le 25 du mois qui
    précède le mois de livraison (si le 25 n'est pas ouvré, on part du jour
    ouvré qui le précède). Les jours fériés sont ceux du calendrier fédéral
    américain, ce qui est une approximation du calendrier NYMEX.
    """
    previous = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=1)
    day_25 = pd.Timestamp(year=previous.year, month=previous.month, day=25)
    holidays = USFederalHolidayCalendar().holidays(f"{previous.year - 1}-01-01", f"{previous.year + 1}-12-31")
    expiry = np.busday_offset(
        day_25.to_datetime64().astype("datetime64[D]"), -3, roll="backward", holidays=holidays.values.astype("datetime64[D]")
    )
    return pd.Timestamp(expiry)


def contract_code(year: int, month: int, root: str = "CL") -> str:
    return f"{root}{MONTH_CODES[month - 1]}{str(year)[-2:]}"


# --------------------------------------------------------------------------
# Entrées / sorties
# --------------------------------------------------------------------------
def load_contracts_csv(path: str | Path) -> pd.DataFrame:
    """Charge un panneau de contrats depuis un CSV (colonnes ``REQUIRED_COLUMNS``)."""
    df = pd.read_csv(path, parse_dates=["date", "expiry"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path} : {sorted(missing)}")
    return df[list(REQUIRED_COLUMNS)].sort_values(["date", "expiry"]).reset_index(drop=True)


def save_contracts_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Source réelle : Yahoo Finance
# --------------------------------------------------------------------------
def download_yahoo(
    start: str = "2021-06-01",
    end: str | None = None,
    life_months: int = 14,
    pause: float = 0.4,
    root: str = "CL",
    exchange: str = "NYM",
) -> pd.DataFrame:
    """Télécharge l'historique des contrats WTI mensuels (``CLZ26.NYM``...) sur Yahoo.

    Nécessite ``yfinance`` et un accès Internet. Les contrats dont l'historique
    est introuvable sont ignorés (avec un avertissement) : la construction de la
    courbe exige seulement ``n_maturities`` contrats vivants à chaque date.

    La fenêtre par défaut commence après avril 2020, mois où le contrat WTI a
    coté à des prix négatifs (le logarithme n'y est pas défini).
    """
    import yfinance as yf  # import tardif : dépendance optionnelle

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for year in range(start_ts.year, end_ts.year + 3):
        for month in range(1, 13):
            expiry = last_trade_date(year, month)
            first_day = expiry - pd.DateOffset(months=life_months)
            if expiry < start_ts or first_day > end_ts:
                continue
            code = contract_code(year, month, root)
            ticker = f"{code}.{exchange}"
            try:
                hist = yf.Ticker(ticker).history(
                    start=max(first_day, start_ts).strftime("%Y-%m-%d"),
                    end=(min(expiry, end_ts) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=False,
                )
            except Exception as exc:  # réseau, limite de débit, ticker inconnu...
                warnings.warn(f"{ticker} : {exc}")
                hist = pd.DataFrame()
            time.sleep(pause)
            if hist.empty or "Close" not in hist:
                missing.append(code)
                continue
            frame = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(hist.index).tz_localize(None).normalize(),
                    "contract": code,
                    "expiry": expiry,
                    "price": hist["Close"].to_numpy(dtype=float),
                }
            )
            rows.append(frame[frame["date"] <= expiry])
    if missing:
        warnings.warn(f"Historique introuvable pour {len(missing)} contrat(s) : {', '.join(missing)}")
    if not rows:
        raise RuntimeError("Aucun contrat téléchargé : vérifiez la connexion ou la version de yfinance.")
    return pd.concat(rows, ignore_index=True).sort_values(["date", "expiry"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Du panneau de contrats à la matrice de rendements
# --------------------------------------------------------------------------
@dataclass
class CurvePanel:
    """Courbe à terme « par rang » : prix et rendements journaliers."""

    prices: pd.DataFrame  # index = dates, colonnes = rang 1..N
    returns: pd.DataFrame  # idem, rendements logarithmiques journaliers
    info: dict = field(default_factory=dict)


def build_curve_panel(
    contracts: pd.DataFrame,
    n_maturities: int = 12,
    max_abs_return: float = 0.25,
) -> CurvePanel:
    """Construit les prix et rendements des ``n_maturities`` premiers contrats.

    Parameters
    ----------
    contracts
        Panneau de contrats (colonnes ``REQUIRED_COLUMNS``).
    n_maturities
        Nombre de maturités conservées (rang 1 = contrat le plus proche).
    max_abs_return
        Les jours où un rendement dépasse ce seuil en valeur absolue sont
        écartés : ce sont presque toujours des erreurs de données.
    """
    df = contracts.loc[contracts["price"] > 0, list(REQUIRED_COLUMNS)]
    wide = df.pivot(index="date", columns="contract", values="price").sort_index()
    expiry = df.groupby("contract")["expiry"].first().sort_values()
    wide = wide[expiry.index]  # colonnes triées par échéance croissante

    dates = wide.index.to_numpy()
    expiry_values = expiry.to_numpy()
    values = wide.to_numpy()
    n_dates, n = len(dates), n_maturities

    ids = np.full((n_dates, n), -1, dtype=int)
    price_grid = np.full((n_dates, n), np.nan)
    for i in range(n_dates):
        alive = np.flatnonzero((expiry_values > dates[i]) & np.isfinite(values[i]))
        if len(alive) >= n:
            ids[i] = alive[:n]
            price_grid[i] = values[i, alive[:n]]

    # Rendement du rang k au jour t = rendement du contrat qui était de rang k la veille.
    returns = np.full((n_dates, n), np.nan)
    for i in range(1, n_dates):
        if ids[i - 1, 0] >= 0:
            j = ids[i - 1]
            returns[i] = np.log(values[i, j] / values[i - 1, j])

    columns = pd.RangeIndex(1, n + 1, name="rang")
    prices_df = pd.DataFrame(price_grid, index=wide.index, columns=columns).dropna()
    returns_df = pd.DataFrame(returns, index=wide.index, columns=columns)

    is_valid = returns_df.notna().all(axis=1)
    is_outlier = returns_df.abs().max(axis=1) > max_abs_return
    kept = returns_df[is_valid & ~is_outlier]
    info = {
        "n_contracts": int(expiry.size),
        "n_maturities": int(n),
        "first_date": str(kept.index.min().date()),
        "last_date": str(kept.index.max().date()),
        "n_returns": int(len(kept)),
        "n_dropped_missing": int((~is_valid).sum() - 1),  # -1 : première date sans rendement
        "n_dropped_outliers": int((is_valid & is_outlier).sum()),
        "max_abs_return": float(max_abs_return),
    }
    return CurvePanel(prices=prices_df.loc[prices_df.index.intersection(kept.index)], returns=kept, info=info)
