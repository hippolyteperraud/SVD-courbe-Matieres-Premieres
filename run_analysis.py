#!/usr/bin/env python
"""SVD de la courbe à terme du WTI (données Yahoo Finance) -> figures (PNG) et chiffres (JSON).

    python run_analysis.py            # télécharge les contrats (quelques minutes) puis trace les figures
    python run_analysis.py --reuse    # réutilise le dernier téléchargement (data/contrats_wti.csv)

Résultats dans ``output/`` : ``figures/*.png`` et ``resultats.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from svdcurve.analysis import run_analysis  # noqa: E402
from svdcurve.data import build_curve_panel, download_yahoo, load_contracts_csv, save_contracts_csv  # noqa: E402
from svdcurve.plots import make_all_figures  # noqa: E402

CACHE = ROOT / "data" / "contrats_wti.csv"
OUT = ROOT / "output"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reuse", action="store_true", help="réutiliser le dernier téléchargement")
    p.add_argument("--start", default="2021-06-01", help="début de l'historique (défaut : 2021-06-01)")
    p.add_argument("--maturities", type=int, default=8, help="nombre d'échéances (défaut : 8)")
    args = p.parse_args()

    if args.reuse and CACHE.exists():
        contracts = load_contracts_csv(CACHE)
    else:
        print("Téléchargement des contrats WTI sur Yahoo Finance (quelques minutes)...")
        contracts = download_yahoo(start=args.start)
        save_contracts_csv(contracts, CACHE)
    print(f"{contracts['contract'].nunique()} contrats.")

    result = run_analysis(build_curve_panel(contracts, n_maturities=args.maturities))
    m = result.metrics
    v = [100 * r for r in m["variance_ratio"][:3]]
    print(f"{m['n_obs']} jours x {m['n_maturities']} échéances. Variance expliquée : {v[0]:.1f} %, {v[1]:.1f} %, {v[2]:.2f} % (3 premiers facteurs).")

    OUT.mkdir(exist_ok=True)
    (OUT / "resultats.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    make_all_figures(result, OUT / "figures")
    print(f"Figures et resultats.json dans {OUT}")


if __name__ == "__main__":
    main()
