# SVD de la courbe à terme du pétrole (WTI)

Application de la **décomposition en valeurs singulières (SVD)** à la courbe à terme du WTI :
8 prix d'échéances qui bougent presque ensemble sont ramenés à **3 facteurs** (niveau, pente, courbure).

Le projet produit, à partir de données de marché :

- les calculs (SVD, variance expliquée, seuil de Gavish–Donoho, reconstruction d'Eckart–Young) ;
- 5 figures + un visuel pour LinkedIn (PNG) ;
- `resultats.json` : tous les chiffres de l'analyse, pour rédiger le rapport à la main.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

## Utilisation

```bash
python run_analysis.py           # télécharge les contrats WTI sur Yahoo Finance (quelques minutes)
python run_analysis.py --reuse   # réutilise le dernier téléchargement (data/contrats_wti.csv)
pytest                           # tests (sur données simulées, hors ligne)
```

Résultats dans `output/` : `figures/` et `resultats.json`. Options : `--start 2021-06-01`, `--maturities 8` (valeur par défaut).

## Méthode en bref

1. Rendements logarithmiques journaliers par **rang** d'échéance (rang 1 = contrat le plus proche). Chaque rendement compare
   le même contrat d'un jour à l'autre : pas de saut artificiel lors du changement de contrat.
2. Centrage, puis SVD économique `B = U Σ Vᵀ`. Les colonnes de `V` sont les facteurs, `UΣ` les scores,
   `σ²/(T−1)` les variances : c'est l'ACP, sans passer par la matrice de covariance.
3. Nombre de facteurs : critère d'énergie (90 % / 99 %) et seuil optimal de Gavish–Donoho.
4. Reconstruction de rang k et vérification du théorème d'Eckart–Young.
5. Identification niveau / pente / courbure par le nombre de changements de signe des coefficients (0, 1, 2),
   puis contrôle par des indicateurs simples.

## Structure

```
run_analysis.py          point d'entrée
src/svdcurve/pca.py      SVD-ACP, Eckart–Young, seuil de Gavish–Donoho
src/svdcurve/data.py     échéances WTI, téléchargement Yahoo Finance, courbe par rang
src/svdcurve/analysis.py indicateurs de l'analyse
src/svdcurve/plots.py    figures
tests/                   tests unitaires
```

## Limites

**Historique Yahoo court.** Yahoo ne conserve pas les contrats expirés : seuls les contrats encore actifs (une quinzaine)
sont téléchargés, donc l'historique commun est court (environ 140 jours pour 8 échéances, 180 pour 6, 100 pour 10).
Le défaut de 8 échéances est un compromis entre le nombre d'échéances et le nombre de jours. Pour un historique plus long,
fournir un export de contrats historiques (colonnes `date, contract, expiry, price`) à `load_contracts_csv`.


L'ACP est linéaire et suppose une covariance stable ; le seuil de Gavish–Donoho suppose un bruit i.i.d. ;
les échéances sont recalculées d'après la règle de la CME (approximation du calendrier NYMEX).
Le téléchargement Yahoo Finance repose sur des tickers par contrat (`CLZ26.NYM`) dont la disponibilité peut varier.
