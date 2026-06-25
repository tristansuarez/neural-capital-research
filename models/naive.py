"""
Benchmark naive: comprar y mantener el activo (buy & hold).

Es la vara de medir. Cualquier modelo direccional sobre un activo debe batir
ESTO fuera de muestra; si no, no aporta nada por bonito que sea su backtest.
Para estrategias neutrales al mercado (como el par cointegrado) el benchmark
relevante no es este, sino "Sharpe distinguible de cero" -> ver validation.py.
"""

from __future__ import annotations
import pandas as pd
from .base import Model


class BuyAndHold(Model):
    name = "buy_and_hold"
    description = ("Comprar y mantener el activo. Es el benchmark: lo que "
                   "obtienes sin modelo ninguno.")

    def __init__(self, asset: str):
        self.asset = asset
        self.assets = [asset]

    def fit(self, train_df: pd.DataFrame) -> None:
        # No hay nada que estimar.
        return None

    def weights(self, history_df: pd.DataFrame) -> dict[str, float]:
        # Siempre 100% largo el activo.
        return {self.asset: 1.0}
