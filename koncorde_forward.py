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

Regla de cada operacion (operando el KONCORDE como dicta el indicador):
  compra a la apertura del dia siguiente a la senal de COMPRA y vende a la
  apertura siguiente a la senal de VENTA (el marron cruza su media a la baja),
  con 0.10% de comision; se compara con meter el mismo dinero en el S&P 500 en
  las MISMAS fechas, para aislar la senal del mercado.
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
    """Forward-test del KONCORDE operado COMO DICTA EL INDICADOR: compra en la
    apertura siguiente a la señal de compra y vende en la apertura siguiente a la
    señal de venta (el marrón cruza su media a la baja). Devuelve trades, curva,
    totales y la lista de operaciones."""
    import escaner_senales_telegram as esc   # reutiliza el cálculo del indicador
    filas = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    señales, vistas = [], set()
    for f in filas:
        clave = (f.get("ticker"), f.get("fecha"))
        if clave in vistas or not all(clave):
            continue
        vistas.add(clave)
        try:
            precio = float(f.get("precio") or 0) or None
        except Exception:
            precio = None
        señales.append({"ticker": f["ticker"], "fecha": f["fecha"], "precio": precio})
    if not señales:
        return [], ([], []), 0, 0, []

    fmin = min(s["fecha"] for s in señales)
    # margen amplio hacia atrás para "calentar" el indicador (ventanas de hasta 90 sesiones)
    desde = (datetime.strptime(fmin, "%Y-%m-%d") - timedelta(days=420)).strftime("%Y-%m-%d")
    idx = _precios("^GSPC", desde)
    idx_open = idx["Open"] if idx is not None else None

    cache, cerradas, ops = {}, [], []
    for s in señales:
        tk = s["ticker"]
        op = {"ticker": tk, "fecha": s["fecha"], "precio": s["precio"],
              "estado": "en observación", "salida": None, "fecha_salida": None, "retorno": None}
        if tk not in cache:
            df = _precios(tk, desde)
            cache[tk] = esc.koncorde(df) if df is not None else None
        kdf = cache[tk]
        if kdf is not None:
            try:
                fechas = kdf.index
                marron = kdf["marron"].values
                media = kdf["media"].values
                pos = fechas.searchsorted(pd.Timestamp(s["fecha"]) + timedelta(days=1))
                if pos < len(kdf):
                    entrada = float(kdf["Open"].iloc[pos]); fent = fechas[pos]
                    # primera señal de VENTA tras la entrada: marrón cruza su media a la baja
                    sell = None
                    for t in range(pos + 1, len(kdf)):
                        a, b, pa, pb = marron[t], media[t], marron[t-1], media[t-1]
                        if np.isfinite(a) and np.isfinite(b) and np.isfinite(pa) and np.isfinite(pb):
                            if a < b and pa >= pb:
                                sell = t; break
                    if sell is not None and sell + 1 < len(kdf):   # se ejecuta la salida a la apertura siguiente
                        salida = float(kdf["Open"].iloc[sell + 1]); fsal = fechas[sell + 1]
                        ret = (salida / entrada - 1.0) * 100 - COSTE
                        ret_idx = None
                        if idx_open is not None:
                            try:
                                ret_idx = (float(idx_open.asof(fsal)) / float(idx_open.asof(fent)) - 1.0) * 100
                            except Exception:
                                ret_idx = None
                        cerradas.append((fsal, ret, ret_idx))
                        op.update({"estado": "cerrada", "salida": round(salida, 2),
                                   "fecha_salida": fsal.strftime("%Y-%m-%d"), "retorno": round(ret, 2)})
                    # si no hay señal de venta todavía, sigue "en observación"
            except Exception:
                pass
        ops.append(op)

    cerradas.sort(key=lambda x: x[0])
    ops.sort(key=lambda o: o["fecha"], reverse=True)
    n_open = len(señales) - len(cerradas)   # todo lo no cerrado sigue en observación
    return cerradas, _curvas(cerradas), len(señales), n_open, ops


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
    return cerradas, _curvas(cerradas), 31, 5, []


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


def _op_cols_koncorde():
    return [{"k": "fecha", "t": "Señal"}, {"k": "ticker", "t": "Valor"},
            {"k": "precio", "t": "Cierre señal"}, {"k": "estado", "t": "Estado"},
            {"k": "fecha_salida", "t": "Salida"},
            {"k": "retorno", "t": "Retorno", "sufijo": "%"}]


def operaciones_plata(csv_path="senal_plata_log.csv"):
    """Lee el log de la señal de plata (par oro-plata en vivo). Vacío si aún no ha disparado."""
    cols = [{"k": "fecha", "t": "Fecha"}, {"k": "z", "t": "z (σ)"},
            {"k": "accion_plata", "t": "Plata"}, {"k": "accion_oro", "t": "Oro"},
            {"k": "ratio_oro_plata", "t": "Ratio O/P"}]
    if not os.path.exists(csv_path):
        return [], cols
    ops = []
    for f in csv.DictReader(open(csv_path, encoding="utf-8")):
        acc_p = (f.get("accion_plata") or "").upper()
        acc_o = "VENDER" if "COMPRAR" in acc_p else ("COMPRAR" if "VENDER" in acc_p else "—")
        ops.append({"fecha": f.get("fecha"), "z": f.get("z"),
                    "accion_plata": acc_p or "—", "accion_oro": acc_o,
                    "ratio_oro_plata": f.get("ratio_oro_plata")})
    ops.sort(key=lambda o: o.get("fecha") or "", reverse=True)
    return ops, cols


def evaluar_koncorde(csv_path=LOG_CSV, sintetico=False):
    if sintetico:
        cerradas, (curva, curva2), n_sen, n_open, ops = _forward_test_sintetico()
    elif os.path.exists(csv_path):
        cerradas, (curva, curva2), n_sen, n_open, ops = _forward_test_real(csv_path)
    else:
        cerradas, curva, curva2, n_sen, n_open, ops = [], [], [], 0, 0, []

    base = {
        "id": "koncorde", "etiqueta": "KONCORDE (S&P 500)",
        "tipo": "Análisis técnico · forward-test en vivo",
        "modelo": "koncorde_v3",
        "descripcion": ("Señal KONCORDE V3 (Blai5 + régimen de Fosback) medida en "
                        "vivo, sin sesgo de supervivencia."),
    }

    if len(cerradas) < 3:
        base.update({
            "sin_datos": True,
            "sin_datos_txt": (
                f"El forward-test tiene {n_open} señal(es) en observación y "
                f"{len(cerradas)} operación(es) cerrada(s). Cada señal se cierra a las "
                f"20 sesiones de registrarse; el veredicto aparecerá cuando se cierren "
                f"unas cuantas. Señales totales registradas: {n_sen}."),
            "cards": [
                {"k": "En observación", "v": str(n_open), "tono": ""},
                {"k": "Operaciones cerradas", "v": str(len(cerradas)), "tono": ""},
                {"k": "Señales totales", "v": str(n_sen), "tono": ""},
            ],
            "operaciones": ops,
            "op_cols": _op_cols_koncorde(),
        })
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
        "diagnostico": {"salida": "marrón cruza su media a la baja", "coste": "0.1%"},
        "curva": curva,
        "curva2": {"nombre": "S&P 500", "datos": curva2},
        "curva_unidad": "€", "curva_base": 0.0,
        "curva_titulo": "Forward-test: estrategia vs S&P 500",
        "curva_sub": f"Beneficio acumulado en papel ({CAPITAL} € por señal), mismas fechas que el índice.",
        "operaciones": ops,
        "op_cols": _op_cols_koncorde(),
    })
    return base
