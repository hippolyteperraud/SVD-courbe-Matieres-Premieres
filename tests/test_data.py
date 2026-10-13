"""Tests de la construction de la courbe à terme."""

import numpy as np
import pandas as pd
import pytest

from simulator import simulate_contracts
from svdcurve.data import build_curve_panel, contract_code, last_trade_date


def test_last_trade_date_matches_cme_calendar():
    # Contrats réels : CLZ25 a cessé de coter le 20/11/2025, CLF26 le 19/12/2025.
    assert last_trade_date(2025, 12) == pd.Timestamp("2025-11-20")
    assert last_trade_date(2026, 1) == pd.Timestamp("2025-12-19")


def test_contract_code():
    assert contract_code(2026, 12) == "CLZ26"
    assert contract_code(2027, 1) == "CLF27"


@pytest.fixture(scope="module")
def contracts() -> pd.DataFrame:
    return simulate_contracts(start="2018-01-02", end="2019-12-31", seed=1)


def test_simulation_schema(contracts):
    assert list(contracts.columns) == ["date", "contract", "expiry", "price"]
    assert (contracts["price"] > 0).all()
    assert (contracts["date"] <= contracts["expiry"]).all()


def test_panel_shapes_and_no_missing(contracts):
    panel = build_curve_panel(contracts, n_maturities=12)
    assert panel.returns.shape[1] == 12
    assert panel.returns.notna().all().all()
    assert panel.info["n_dropped_outliers"] == 0
    assert panel.prices.shape[1] == 12


def test_returns_are_computed_contract_by_contract():
    """Un déplacement parallèle de la courbe donne le même rendement à tous les rangs,
    y compris le jour du roulement (pas de faux saut F2/F1)."""
    dates = pd.bdate_range("2020-01-02", periods=60)
    rows = []
    shift = np.where(np.arange(60) % 2 == 0, 0.01, -0.005)
    level = np.exp(np.cumsum(shift))
    for m in range(1, 8):
        expiry = dates[0] + pd.Timedelta(days=30 * m + 3)
        base = 50 * (1 + 0.01 * m)  # contango constant
        mask = dates <= expiry
        rows.append(
            pd.DataFrame({"date": dates[mask], "contract": f"C{m}", "expiry": expiry, "price": base * level[mask]})
        )
    panel = build_curve_panel(pd.concat(rows), n_maturities=3, max_abs_return=0.5)
    spread = panel.returns.max(axis=1) - panel.returns.min(axis=1)
    assert spread.max() < 1e-12
