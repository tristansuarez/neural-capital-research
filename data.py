"""
Capa de datos del laboratorio.

Descarga precios de cierre diarios. Fuente primaria Yahoo Finance (yfinance);
si falla, respaldo en Stooq (descarga CSV directa, sin clave). Cachea en disco
para no depender de la red en cada ejecucion.

Nota: en GitHub Actions hay salida a internet libre, asi que aqui funcionara.
En entornos con red restringida usa cargar_sinteticos() para probar el motor.
"""

from __future__ import annotations
import io
import os
import datetime as dt
import numpy as np
import pandas as pd

import config


def _desde_yahoo(ticker: str, anos: int) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        fin = dt.date.today()
        ini = fin - dt.timedelta(days=int(anos * 365.25) + 5)
        df = yf.download(ticker, start=ini.isoformat(), end=fin.isoformat(),
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        col = "Close" if "Close" in df.columns else df.columns[0]
        s = df[col].copy()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception:
        return None


def _desde_stooq(ticker: str) -> pd.Series | None:
    try:
        import urllib.request
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        if "Close" not in df.columns or df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")["Close"].dropna()
    except Exception:
        return None


def cargar_activo(nombre: str, anos: int | None = None) -> pd.Series:
    """Carga un activo (cache -> Yahoo -> Stooq)."""
    anos = anos or config.ANOS_HISTORICO
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache = os.path.join(config.CACHE_DIR, f"{nombre}.csv")

    cfg = config.ACTIVOS[nombre]
    s = _desde_yahoo(cfg["yahoo"], anos)
    if s is None:
        s = _desde_stooq(cfg["stooq"])
    if s is None and os.path.exists(cache):
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    if s is None:
        raise RuntimeError(f"No se pudo obtener datos de '{nombre}'")

    s = s.sort_index()
    s.name = nombre
    s.to_frame().to_csv(cache)
    return s


def cargar_panel(activos: list[str], anos: int | None = None) -> pd.DataFrame:
    """Carga varios activos alineados por fecha (solo dias comunes)."""
    series = [cargar_activo(a, anos) for a in activos]
    df = pd.concat(series, axis=1).dropna()
    df.columns = activos
    return df


def cargar_sinteticos(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """
    Genera un par oro-plata COINTEGRADO sintetico para probar el motor en
    entornos sin red. La respuesta correcta es conocida: existe reversion.
    """
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range("2013-01-01", periods=n)
    # plata: paseo aleatorio en logs
    log_plata = np.cumsum(rng.normal(0, 0.012, n)) + np.log(23)
    # spread estacionario (OU) que ata el oro a la plata
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.97 * spread[t-1] + rng.normal(0, 0.01)
    beta = 1.0
    log_oro = np.log(80) + beta * log_plata + spread  # oro cointegrado con plata
    df = pd.DataFrame({"oro": np.exp(log_oro), "plata": np.exp(log_plata)},
                      index=fechas)
    return df
