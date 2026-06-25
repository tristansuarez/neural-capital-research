"""
koncorde_forward.py
===================
Convierte el forward-test del KONCORDE (su senales_log.csv en vivo) en un
experimento del laboratorio, con el mismo esquema que los demas modelos.

Por que forward-test y no backtest historico: backtestear el indicador sobre
los componentes ACTUALES del S&P 500 introduce sesgo de supervivencia (solo
pruebas las empresas que sobrevivieron hasta hoy), que infla los resultados. El
forward-test registra las senales en el momento, sin esa trampa. Por eso es la
forma honesta de medir el KONCORDE; su contrapartida es que el tamano muestral
crece despacio, con el tiempo.

Regla de cada operacion (igual que el generar_web original):
  compra a la apertura del dia siguiente a la senal, vende a 20 sesiones, con
  0.10% de comision; se compara con meter el mismo dinero en el S&P 500 en las
  MISMAS fechas, para aislar la senal del mercado.
"""

from __future__ import annotations
import csv
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CAPITAL = 1000       # € imaginarios por senal
HORIZONTE = 20       # sesiones hasta la venta
COSTE = 0.10         # % comision ida y vuelta
LOG_CSV = "senales_log.csv"


def _bootstrap_mean(x, n_boot=2000, seed=7):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) < 3:
        return {"p_valor": 1.0, "ic90": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    p = float(np.mean(means <= 0))
    lo, hi = np.percentile(means, [5, 95])
    return {"p_valor": round(p, 4), "ic90": [round(float(lo), 3), round(float(hi), 3)]}


def _precios(ticker, desde):
    import yfinance as yf
    try:
        df = yf.download(ticker, start=desde, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception:
        return None


def _forward_test_real(csv_path):
    """Replica la logica del forward-test sobre el CSV real. Devuelve trades+curva."""
    filas = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    señales, vistas = [], set()
    for f in filas:
        clave = (f.get("ticker"), f.get("fecha"))
        if clave in vistas or not all(clave):
            continue
        vistas.add(clave)
        señales.append({"ticker": f["ticker"], "fecha": f["fecha"]})
    if not señales:
        return [], [], 0, 0

    fmin = min(s["fecha"] for s in señales)
    desde = (datetime.strptime(fmin, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    idx = _precios("^GSPC", desde)
    idx_open = idx["Open"] if idx is not None else None

    cache, cerradas, n_abiertas = {}, [], 0
    for s in señales:
        tk = s["ticker"]
        if tk not in cache:
            cache[tk] = _precios(tk, desde)
        df = cache[tk]
        if df is None:
            continue
        try:
            fechas = df.index
            pos = fechas.searchsorted(pd.Timestamp(s["fecha"]) + timedelta(days=1))
            if pos >= len(df):
                continue
            entrada = float(df["Open"].iloc[pos]); fent = fechas[pos]
            if pos + HORIZONTE < len(df):
                salida = float(df["Open"].iloc[pos + HORIZONTE]); fsal = fechas[pos + HORIZONTE]
                ret = (salida / entrada - 1.0) * 100 - COSTE
                ret_idx = None
                if idx_open is not None:
                    try:
                        ret_idx = (float(idx_open.asof(fsal)) / float(idx_open.asof(fent)) - 1.0) * 100
                    except Exception:
                        ret_idx = None
                cerradas.append((fsal, ret, ret_idx))
            else:
                n_abiertas += 1
        except Exception:
            pass

    cerradas.sort(key=lambda x: x[0])
    return cerradas, _curvas(cerradas), len(señales), n_abiertas


def _forward_test_sintetico(seed=3):
    """Operaciones ficticias para probar el motor sin red: exceso centrado en ~0."""
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 10)
    cerradas = []
    for i in range(26):
        fsal = base + timedelta(days=14 * i)
        ret_idx = float(rng.normal(0.5, 3.0))
        ret = ret_idx + float(rng.normal(-0.1, 4.0))  # sin ventaja real sobre el indice
        cerradas.append((pd.Timestamp(fsal), ret, ret_idx))
    return cerradas, _curvas(cerradas), 31, 5


def _curvas(cerradas):
    """Curva de beneficio acumulado en € para estrategia e indice."""
    curva, curva2 = [], []
    acum_e = acum_i = 0.0
    for fsal, re, ri in cerradas:
        acum_e += CAPITAL * re / 100
        if ri is not None:
            acum_i += CAPITAL * ri / 100
        f = fsal.strftime("%Y-%m-%d")
        curva.append({"fecha": f, "valor": round(acum_e, 1)})
        curva2.append({"fecha": f, "valor": round(acum_i, 1)})
    return curva, curva2


def evaluar_koncorde(csv_path=LOG_CSV, sintetico=False):
    if sintetico:
        cerradas, (curva, curva2), n_sen, n_open = _forward_test_sintetico()
    elif os.path.exists(csv_path):
        cerradas, (curva, curva2), n_sen, n_open = _forward_test_real(csv_path)
    else:
        cerradas, curva, curva2, n_sen, n_open = [], [], [], 0, 0

    base = {
        "id": "koncorde", "etiqueta": "KONCORDE (S&P 500)",
        "tipo": "Análisis técnico · forward-test en vivo",
        "modelo": "koncorde_v3",
        "descripcion": ("Señal KONCORDE V3 (Blai5 + régimen de Fosback) medida en "
                        "vivo, sin sesgo de supervivencia."),
    }

    if len(cerradas) < 3:
        base.update({"sin_datos": True,
                     "sin_datos_txt": (f"El forward-test lleva {len(cerradas)} operaciones "
                                       "cerradas. Necesita unas cuantas más para un veredicto; "
                                       "se irá llenando solo con cada señal.")})
        return base

    rets = [c[1] for c in cerradas]
    rets_idx = [c[2] for c in cerradas if c[2] is not None]
    exceso = [c[1] - c[2] for c in cerradas if c[2] is not None]
    mean_exc = float(np.mean(exceso)) if exceso else 0.0
    sig = _bootstrap_mean(exceso)

    def pct(x): return f"{x:+.2f}%"
    base.update({
        "headline": {"valor": round(mean_exc, 2),
                     "etiqueta": "Exceso medio por operación vs S&P 500",
                     "sufijo": "%", "decimales": 2},
        "significancia": {"p_valor": sig["p_valor"], "ic90": sig["ic90"],
                          "etiqueta": "exceso medio (%)"},
        "cards": [
            {"k": "Operaciones cerradas", "v": str(len(cerradas)), "tono": ""},
            {"k": "Aciertos", "v": f"{100*np.mean([1 if r>0 else 0 for r in rets]):.0f}%", "tono": ""},
            {"k": "Retorno medio estrategia", "v": pct(float(np.mean(rets))),
             "tono": "pos" if np.mean(rets) >= 0 else "neg"},
            {"k": "Retorno medio S&P 500", "v": pct(float(np.mean(rets_idx))) if rets_idx else "—", "tono": ""},
            {"k": "En observación", "v": str(n_open), "tono": ""},
            {"k": "Señales totales", "v": str(n_sen), "tono": ""},
        ],
        "diagnostico": {"horizonte": "20 sesiones", "coste": "0.1%"},
        "curva": curva,
        "curva2": {"nombre": "S&P 500", "datos": curva2},
        "curva_unidad": "€", "curva_base": 0.0,
        "curva_titulo": "Forward-test: estrategia vs S&P 500",
        "curva_sub": f"Beneficio acumulado en papel ({CAPITAL} € por señal), mismas fechas que el índice.",
    })
    return base
