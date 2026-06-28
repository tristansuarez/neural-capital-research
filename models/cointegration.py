"""
Modelo de par cointegrado (reversion a la media) generico para dos activos.

Idea (arbitraje estadistico clasico): si dos activos comparten drivers, el
"spread" entre sus log-precios tiende a ser estacionario. Cuando el spread se
aleja mucho de su media, apostamos a que vuelve. Es la unica familia de hipotesis
del laboratorio con fundamento teorico defendible.

Honestidad por delante:
  - Si el test de cointegracion NO se cumple en la ventana de entrenamiento, el
    modelo se queda FUERA (pesos 0). No inventamos una relacion que no esta.
  - La posicion es continua (proporcional a la desviacion), sin estado oculto,
    para que el walk-forward sea limpio y reproducible.

Definiciones (a = activo 1, b = activo 2):
  spread = log(a) - beta * log(b) - alpha      (beta = ratio de cobertura)
  z      = (spread - media) / desviacion       (normalizado en la ventana)
  z alto -> spread caro -> CORTOS del spread (corto a, largo b), y viceversa.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

from .base import Model


class PairsModel(Model):
    """Par cointegrado generico. Por defecto, oro-plata."""

    def __init__(self, a: str = "oro", b: str = "plata",
                 coint_pvalue_max: float = 0.10, z_entry: float = 2.0):
        self.a = a
        self.b = b
        self.assets = [a, b]
        self.name = f"par_{a}_{b}"
        self.description = (f"Par cointegrado {a}-{b}. Opera la reversion a la media "
                            f"del spread cuando el test de cointegracion lo respalda.")
        self.coint_pvalue_max = coint_pvalue_max
        self.z_entry = z_entry
        self.beta = None
        self.alpha = None
        self.spread_mean = None
        self.spread_std = None
        self.cointegrated = False
        self.coint_pvalue = None
        self.half_life = None

    # --- entrenamiento -------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> None:
        a = np.log(train_df[self.a].astype(float))
        b = np.log(train_df[self.b].astype(float))

        try:
            _, pvalue, _ = coint(a, b)
        except Exception:
            pvalue = 1.0
        self.coint_pvalue = float(pvalue)
        self.cointegrated = bool(pvalue <= self.coint_pvalue_max)

        X = sm.add_constant(b.values)
        ols = sm.OLS(a.values, X).fit()
        self.alpha = float(ols.params[0])
        self.beta = float(ols.params[1])

        spread = a - self.beta * b - self.alpha
        self.spread_mean = float(spread.mean())
        self.spread_std = float(spread.std(ddof=1))
        if self.spread_std == 0 or np.isnan(self.spread_std):
            self.spread_std = 1e-9

        self.half_life = self._half_life(spread)

    @staticmethod
    def _half_life(spread: pd.Series) -> float | None:
        s = spread.dropna()
        lag = s.shift(1).dropna()
        delta = (s - s.shift(1)).dropna()
        lag = lag.loc[delta.index]
        if len(lag) < 10:
            return None
        X = sm.add_constant(lag.values)
        bcoef = sm.OLS(delta.values, X).fit().params[1]
        if bcoef >= 0:
            return None
        return float(-np.log(2) / np.log(1 + bcoef))

    # --- senal ---------------------------------------------------------------
    def weights(self, history_df: pd.DataFrame) -> dict[str, float]:
        if not self.cointegrated or self.beta is None:
            return {self.a: 0.0, self.b: 0.0}

        a_now = np.log(float(history_df[self.a].iloc[-1]))
        b_now = np.log(float(history_df[self.b].iloc[-1]))
        spread_now = a_now - self.beta * b_now - self.alpha
        z = (spread_now - self.spread_mean) / self.spread_std

        pos_spread = -np.clip(z / self.z_entry, -1.0, 1.0)
        w_a = pos_spread
        w_b = -pos_spread * self.beta
        gross = abs(w_a) + abs(w_b)
        if gross > 0:
            w_a /= gross
            w_b /= gross
        return {self.a: float(w_a), self.b: float(w_b)}


# Compatibilidad: el par oro-plata original.
GoldSilverPairs = PairsModel
