"""
Modelo de par cointegrado oro-plata con reversion a la media.

Idea (arbitraje estadistico clasico): el oro y la plata comparten drivers
(tipos reales, dolar, aversion al riesgo), asi que el "spread" entre sus
log-precios tiende a ser estacionario. Cuando el spread se aleja mucho de su
media, apostamos a que vuelve. Es la unica hipotesis del laboratorio con
fundamento teorico defendible.

Honestidad por delante:
  - Si el test de cointegracion NO se cumple en la ventana de entrenamiento,
    el modelo se queda FUERA (pesos 0). No inventamos una relacion que no esta.
  - La posicion es continua (proporcional a la desviacion), sin estado oculto,
    para que el walk-forward sea limpio y reproducible.

Definiciones:
  spread = log(oro) - beta * log(plata) - alpha      (beta = ratio de cobertura)
  z      = (spread - media) / desviacion             (normalizado en la ventana)
  Si z es alto -> spread caro -> nos ponemos CORTOS del spread
                  (corto oro, largo plata), y viceversa.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

from .base import Model


class GoldSilverPairs(Model):
    name = "par_oro_plata"
    description = ("Par cointegrado oro-plata. Opera la reversion a la media "
                   "del spread cuando el test de cointegracion lo respalda.")
    assets = ["oro", "plata"]

    def __init__(self, coint_pvalue_max: float = 0.10, z_entry: float = 2.0):
        # Umbral del p-valor para considerar el par cointegrado.
        self.coint_pvalue_max = coint_pvalue_max
        # Desviacion (en z) a la que la posicion alcanza su maximo (+-1).
        self.z_entry = z_entry
        # Parametros estimados en cada fit():
        self.beta = None
        self.alpha = None
        self.spread_mean = None
        self.spread_std = None
        self.cointegrated = False
        self.coint_pvalue = None
        self.half_life = None  # dias hasta corregir media desviacion (diagnostico)

    # --- entrenamiento -------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> None:
        oro = np.log(train_df["oro"].astype(float))
        plata = np.log(train_df["plata"].astype(float))

        # 1) Test de cointegracion de Engle-Granger.
        try:
            _, pvalue, _ = coint(oro, plata)
        except Exception:
            pvalue = 1.0
        self.coint_pvalue = float(pvalue)
        self.cointegrated = bool(pvalue <= self.coint_pvalue_max)

        # 2) Ratio de cobertura por MCO: log(oro) = alpha + beta*log(plata).
        X = sm.add_constant(plata.values)
        ols = sm.OLS(oro.values, X).fit()
        self.alpha = float(ols.params[0])
        self.beta = float(ols.params[1])

        # 3) Spread y su normalizacion en la ventana de entrenamiento.
        spread = oro - self.beta * plata - self.alpha
        self.spread_mean = float(spread.mean())
        self.spread_std = float(spread.std(ddof=1))
        if self.spread_std == 0 or np.isnan(self.spread_std):
            self.spread_std = 1e-9

        # 4) Vida media de la reversion (OU via AR(1) sobre el spread).
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
        b = sm.OLS(delta.values, X).fit().params[1]
        if b >= 0:
            return None  # no hay reversion
        return float(-np.log(2) / np.log(1 + b))

    # --- senal ---------------------------------------------------------------
    def weights(self, history_df: pd.DataFrame) -> dict[str, float]:
        if not self.cointegrated or self.beta is None:
            return {"oro": 0.0, "plata": 0.0}

        oro_now = np.log(float(history_df["oro"].iloc[-1]))
        plata_now = np.log(float(history_df["plata"].iloc[-1]))
        spread_now = oro_now - self.beta * plata_now - self.alpha
        z = (spread_now - self.spread_mean) / self.spread_std

        # Posicion sobre el spread: continua y acotada en [-1, 1].
        # z alto -> spread caro -> corto el spread (pos negativa).
        pos_spread = -np.clip(z / self.z_entry, -1.0, 1.0)

        # Traducir a pesos por pata. Corto del spread = corto oro + largo plata.
        w_oro = pos_spread
        w_plata = -pos_spread * self.beta
        # Normalizar la exposicion bruta a ~1 para que los costes sean comparables
        gross = abs(w_oro) + abs(w_plata)
        if gross > 0:
            w_oro /= gross
            w_plata /= gross
        return {"oro": float(w_oro), "plata": float(w_plata)}
