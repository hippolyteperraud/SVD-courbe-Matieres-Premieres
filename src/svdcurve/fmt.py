"""Mise en forme des nombres à la française."""

from __future__ import annotations

import math

FRENCH_MONTHS = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)  # fmt: skip


def fr(x: float, digits: int = 1) -> str:
    """Nombre avec virgule décimale : ``fr(89.74) -> '89,7'``."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "\u2212")


def sci_tex(x: float, digits: int = 1) -> str:
    """Notation scientifique pour LaTeX : ``2,7 \\times 10^{-14}`` (en mode texte)."""
    if x == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10**exponent
    return f"${fr(mantissa, digits).replace(',', '{,}')}\\times 10^{{{exponent}}}$"


def french_date(year: int, month: int) -> str:
    return f"{FRENCH_MONTHS[month - 1].capitalize()} {year}"
