"""Graphiques du rapport (Matplotlib).

Palette : trois teintes catégorielles validées (bleu, orange, aqua) pour les
trois facteurs, une rampe bleue pour les rangs de reconstruction, encre
neutre pour le texte. Les étiquettes sont écrites en encre (jamais dans la
couleur de la série) et chaque série porte aussi un marqueur distinct.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .analysis import FACTOR_NAMES, AnalysisResult  # noqa: E402
from .fmt import fr  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FACTOR_COLORS = {"Niveau": "#2a78d6", "Pente": "#eb6834", "Courbure": "#1baf7a"}
FACTOR_MARKERS = {"Niveau": "o", "Pente": "s", "Courbure": "^"}
RAMP_BLUE = {1: "#86b6ef", 2: "#3987e5", 3: "#184f95"}
NEUTRAL_BAR = "#c9d3e0"


def _fr_formatter(digits: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda v, _: fr(v, digits))


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#b9b8b3",
            "axes.labelcolor": INK2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def _finish(fig, result: AnalysisResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _direct_labels(ax, x: float, items: list[tuple[float, str, str]], dx: float, gap_fraction: float = 0.075) -> None:
    """Étiquettes à droite des courbes, décalées si besoin, reliées par un petit trait."""
    ymin, ymax = ax.get_ylim()
    gap = gap_fraction * (ymax - ymin)
    ordered = sorted(items, key=lambda item: item[0])
    placed: list[float] = []
    for y, _, _ in ordered:
        placed.append(y if not placed else max(y, placed[-1] + gap))
    for (y, text, _), y_label in zip(ordered, placed):
        ax.plot([x + 0.12 * dx, x + 0.45 * dx], [y, y_label], color=INK2, linewidth=0.8, clip_on=False)
        ax.text(x + 0.55 * dx, y_label, text, va="center", ha="left", color=INK, fontsize=9, clip_on=False)


def _tight(fig) -> None:
    """Mise en page serrée en gardant une marge basse pour le filigrane éventuel."""
    fig.tight_layout(rect=(0, 0.04, 1, 1))


# --------------------------------------------------------------------------
def figure_curves(result: AnalysisResult, path: Path) -> Path:
    """Figure 1 : trois formes de courbe observées (backwardation, médiane, contango)."""
    _setup_style()
    panel, shape = result.panel, result.metrics["curve_shape"]
    ranks = np.arange(1, panel.prices.shape[1] + 1)
    slopes = shape["slope_pct"]  # ln(prix rang N / prix rang 1), en %
    # « Contango » seulement si la courbe est réellement croissante ; sinon c'est la courbe la plus plate.
    flat_label = "Contango" if slopes["contango"] > 0 else "Courbe la plus plate"
    steep_label = "Backwardation" if slopes["backwardation"] < 0 else "Pente la plus basse"
    spec = [
        ("contango", flat_label, FACTOR_COLORS["Niveau"], "o"),
        ("median", "Pente médiane", INK2, "s"),
        ("backwardation", steep_label, FACTOR_COLORS["Pente"], "^"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    items = []
    for key, label, color, marker in spec:
        date = shape["dates"][key]
        y = panel.prices.loc[date].to_numpy()
        ax.plot(ranks, y, color=color, marker=marker, label=f"{label} ({date})")
        items.append((y[-1], f"{label}\n{date}", color))
    ax.set_xlim(0.7, ranks[-1] + 3.4)
    ax.set_xticks(ranks)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Rang du contrat (1 = échéance la plus proche, environ un mois d'écart par rang)")
    ax.set_ylabel("Prix à terme (USD par baril)")
    ax.set_title("Une courbe à terme est un vecteur de prix : trois formes observées")
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.45, -0.17))
    fig.canvas.draw()
    _direct_labels(ax, ranks[-1], items, dx=0.45, gap_fraction=0.12)
    _tight(fig)
    return _finish(fig, result, path)


def figure_variance(result: AnalysisResult, path: Path) -> Path:
    """Figure 2 : variance expliquée et valeurs singulières avec le seuil optimal."""
    _setup_style()
    m = result.metrics
    ratio = 100 * np.array(m["variance_ratio"])
    cumulative = 100 * np.array(m["variance_cumulative"])
    s = np.array(m["singular_values"])
    k_gd, tau = m["gavish_donoho"]["rank"], m["gavish_donoho"]["tau"]
    idx = np.arange(1, len(ratio) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [FACTOR_COLORS["Niveau"] if i <= k_gd else NEUTRAL_BAR for i in idx]
    ax1.bar(idx, ratio, width=0.62, color=colors, label="Part de variance de la composante")
    ax1.plot(idx, cumulative, color=INK, marker="o", markersize=4.5, linewidth=1.6, label="Part cumulée")
    ax1.annotate(f"{fr(ratio[0])} %", (idx[0], ratio[0] / 2), textcoords="offset points", xytext=(16, 0), ha="left", va="center", fontsize=9, color=INK)
    for i in (1, 2):
        ax1.annotate(f"{fr(ratio[i])} %", (idx[i], ratio[i]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=INK)
    ax1.set_ylim(0, 108)
    ax1.set_xticks(idx)
    ax1.grid(axis="x", visible=False)
    ax1.set_xlabel("Composante principale")
    ax1.set_ylabel("Part de la variance totale (%)")
    ax1.yaxis.set_major_formatter(_fr_formatter())
    ax1.set_title("Variance expliquée par composante")
    ax1.legend(loc="center right")

    above = idx <= k_gd
    ax2.semilogy(idx[above], s[above], "o", color=FACTOR_COLORS["Niveau"], label="Retenues (au-dessus du seuil)")
    ax2.semilogy(idx[~above], s[~above], "o", color="#8c97a6", label="Écartées (bruit)")
    ax2.axhline(tau, color=INK, linestyle="--", linewidth=1.4)
    ax2.annotate(f"seuil τ = {fr(tau, 4)}", (len(idx), tau), textcoords="offset points", xytext=(0, 5), ha="right", fontsize=9, color=INK)
    ax2.set_xticks(idx)
    ax2.grid(axis="x", visible=False)
    ax2.set_xlabel("Indice k de la valeur singulière")
    ax2.set_ylabel("Valeur singulière σₖ (échelle log)")
    ax2.set_title("Seuil dur optimal de Gavish–Donoho")
    ax2.legend(loc="upper right")
    _tight(fig)
    return _finish(fig, result, path)


def figure_factors(result: AnalysisResult, path: Path) -> Path:
    """Figure 3 : les trois premières composantes (niveau, pente, courbure)."""
    _setup_style()
    m = result.metrics
    loadings = np.array(m["loadings"])
    ranks = np.arange(1, loadings.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    items = []
    for i, name in enumerate(FACTOR_NAMES):
        share = 100 * m["variance_ratio"][i]
        ax.plot(ranks, loadings[i], color=FACTOR_COLORS[name], marker=FACTOR_MARKERS[name], label=f"{name} ({fr(share)} %)")
        items.append((loadings[i, -1], name, FACTOR_COLORS[name]))
    ax.axhline(0, color="#b9b8b3", linewidth=1.0)
    ax.set_xlim(0.7, ranks[-1] + 2.6)
    ax.set_xticks(ranks)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Rang du contrat (1 = échéance la plus proche)")
    ax.set_ylabel("Coefficient de la composante")
    ax.yaxis.set_major_formatter(_fr_formatter(1))
    ax.set_title("Les trois facteurs de la courbe : niveau, pente, courbure")
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.45, -0.17))
    fig.canvas.draw()
    _direct_labels(ax, ranks[-1], items, dx=0.4, gap_fraction=0.08)
    _tight(fig)
    return _finish(fig, result, path)


def figure_reconstruction(result: AnalysisResult, path: Path) -> Path:
    """Figure 4 : erreur de reconstruction en fonction du rang et exemple d'une journée."""
    _setup_style()
    m = result.metrics
    rec = m["reconstruction"]
    ks = np.array(rec["k"][:-1])  # le rang complet (erreur nulle) n'a pas de sens en échelle log
    rmse = np.array(rec["rmse_bp_per_day"][:-1])
    ex = m["example_day"]
    n_mat = m["n_maturities"]
    ranks = np.arange(1, n_mat + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    ax1.semilogy(ks, rmse, color=INK, marker="o", markersize=4.5, linewidth=1.6)
    for k in (1, 2, 3):
        ax1.plot([k], [rmse[k]], marker="o", markersize=8, color=RAMP_BLUE[k], markeredgecolor=INK, markeredgewidth=0.8, zorder=3)
        ax1.annotate(f"{fr(rmse[k])} pb", (k, rmse[k]), textcoords="offset points", xytext=(7, 6), fontsize=9, color=INK)
    ax1.set_xticks(ks)
    ax1.grid(axis="x", visible=False)
    ax1.set_xlabel("Nombre de facteurs conservés k")
    ax1.set_ylabel("Erreur quadratique moyenne (points de base / jour)")
    ax1.set_title("Eckart–Young : erreur selon le rang")

    ax2.plot(ranks, 100 * np.array(ex["observed"]), color=INK, marker="o", linewidth=1.2, label="Observé")
    for k in (1, 2, 3):
        ax2.plot(ranks, 100 * np.array(ex[f"k{k}"]), color=RAMP_BLUE[k], linewidth=2.0, label=f"{k} facteur{'s' if k > 1 else ''}")
    ax2.axhline(0, color="#b9b8b3", linewidth=1.0)
    ax2.set_xticks(ranks)
    ax2.grid(axis="x", visible=False)
    ax2.set_xlabel("Rang du contrat")
    ax2.set_ylabel("Rendement du jour (%)")
    ax2.yaxis.set_major_formatter(_fr_formatter(1))
    ax2.set_title(f"Journée la plus agitée ({ex['date']})")
    ax2.legend(loc="best", ncols=2)
    _tight(fig)
    return _finish(fig, result, path)


def figure_factor_volatility(result: AnalysisResult, path: Path, window: int = 60) -> Path:
    """Figure 5 : volatilité annualisée glissante de chaque facteur."""
    _setup_style()
    window = max(5, min(window, len(result.scores) // 3))  # fenêtre adaptée si l'historique est court
    vol = 100 * result.scores.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    items = []
    for name in vol.columns:
        series = vol[name].dropna()
        ax.plot(series.index, series, color=FACTOR_COLORS[name], label=name)
        items.append((float(series.iloc[-1]), name, FACTOR_COLORS[name]))
    ax.set_yscale("log")
    ax.set_ylabel("Volatilité annualisée (%, échelle log)")
    ax.set_title(f"Volatilité de chaque facteur sur {window} jours glissants")
    ax.set_yticks([5, 10, 20, 50, 100, 200], labels=["5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.45, -0.12))
    xmin, xmax = ax.get_xlim()
    pad = (xmax - xmin) * 0.12
    ax.set_xlim(xmin, xmax + pad)
    fig.canvas.draw()
    for y, text, _ in sorted(items):
        ax.text(xmax + 0.1 * pad, y, text, va="center", fontsize=9, color=INK)
    _tight(fig)
    return _finish(fig, result, path)


def figure_linkedin(result: AnalysisResult, path: Path) -> Path:
    """Visuel unique pour LinkedIn (1200 x 675 px) : titre, facteurs et variance expliquée."""
    _setup_style()
    plt.rcParams.update({"font.size": 12, "legend.fontsize": 11})
    m = result.metrics
    loadings = np.array(m["loadings"])
    ranks = np.arange(1, loadings.shape[1] + 1)
    top3 = 100 * sum(m["variance_ratio"][:3])
    shares = [100 * r for r in m["variance_ratio"][:3]] + [100 - top3]

    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.text(0.05, 0.93, f"3 facteurs expliquent {fr(top3)} % des mouvements", fontsize=23, weight="bold", color=INK, va="top")
    fig.text(0.05, 0.865, "de la courbe à terme du pétrole WTI (12 échéances), par décomposition en valeurs singulières", fontsize=13, color=INK2, va="top")
    ax = fig.add_axes([0.07, 0.17, 0.52, 0.60])
    items = []
    for i, name in enumerate(FACTOR_NAMES):
        ax.plot(ranks, loadings[i], color=FACTOR_COLORS[name], marker=FACTOR_MARKERS[name], linewidth=2.6, markersize=7)
        items.append((loadings[i, -1], name, FACTOR_COLORS[name]))
    ax.axhline(0, color="#b9b8b3", linewidth=1.0)
    ax.set_xlim(0.7, ranks[-1] + 3.0)
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Échéance du contrat (1 = la plus proche)")
    ax.set_ylabel("Coefficient du facteur")
    ax.yaxis.set_major_formatter(_fr_formatter(1))
    fig.canvas.draw()
    _direct_labels(ax, ranks[-1], items, dx=0.45, gap_fraction=0.09)

    bx = fig.add_axes([0.73, 0.17, 0.24, 0.60])
    labels = ["Niveau", "Pente", "Courbure", "Reste"]
    colors = [FACTOR_COLORS["Niveau"], FACTOR_COLORS["Pente"], FACTOR_COLORS["Courbure"], NEUTRAL_BAR]
    y = np.arange(len(labels))[::-1]
    bx.barh(y, shares, height=0.58, color=colors)
    for yi, share in zip(y, shares):
        bx.text(share + 1.5, yi, f"{fr(share)} %", va="center", fontsize=12, color=INK)
    bx.set_yticks(y, labels)
    bx.set_xlim(0, 118)
    bx.set_xticks([])
    bx.grid(visible=False)
    bx.spines[["left", "bottom"]].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_title("Part de la variance", fontsize=12)
    data = m["data"]
    fig.text(0.05, 0.03, f"{m['n_obs']} rendements journaliers, {data['first_date']} → {data['last_date']}. Source des prix : Yahoo Finance.", fontsize=10, color=INK2)
    return _finish(fig, result, path)


def make_all_figures(result: AnalysisResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "curves": figure_curves(result, out_dir / "fig1_courbes.png"),
        "variance": figure_variance(result, out_dir / "fig2_variance.png"),
        "factors": figure_factors(result, out_dir / "fig3_facteurs.png"),
        "reconstruction": figure_reconstruction(result, out_dir / "fig4_reconstruction.png"),
        "volatility": figure_factor_volatility(result, out_dir / "fig5_volatilite.png"),
        "linkedin": figure_linkedin(result, out_dir / "linkedin_visuel.png"),
    }
