"""
combinaciones.py — Exprimir lo que ha funcionado, con FDR sobre TODO lo probado.
===============================================================================
Dos frentes, ambos con corrección conjunta por multiple-testing:

  A) ESTACIONALIDAD DEL ORO con historia larga. R1 (rotar al oro en agosto-
     septiembre) dio +1,4 puntos anuales pero con solo 20 años y 40 rotaciones:
     insuficiente. Aquí se testea con ~50 años de oro (desde 1975) y se prueban
     varias ventanas de meses, con FDR conjunto.

  B) COMBINACIONES DEFENSIVAS. Lo único que ha aportado en todo el laboratorio es
     la gestión de riesgo (cartera permanente, rotación defensiva, momentum de
     serie temporal). Aquí se combinan entre sí para ver si se suman o se estorban.

AVISO METODOLÓGICO: probar muchas combinaciones garantiza encontrar «ganadoras»
por azar. Por eso el FDR se aplica al CONJUNTO de todas las probadas, y se reporta
cuántas se probaron. Una combinación que solo destaca sin corregir no vale nada.
NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

COSTE = 0.05
VENTANAS_MES = {
    "ago-sep": [8, 9],
    "ago-oct": [8, 9, 10],
    "may-oct": [5, 6, 7, 8, 9, 10],
    "sep": [9],
}


def _cargar_largo(sintetico=False):
    """Oro y bolsa con la máxima historia disponible."""
    if sintetico:
        rng = np.random.default_rng(53)
        n = 600
        idx = pd.date_range("1976-01-31", periods=n, freq="ME")
        return pd.DataFrame({
            "bolsa": 100 * np.exp(np.cumsum(rng.normal(0.007, 0.042, n))),
            "oro": 100 * np.exp(np.cumsum(rng.normal(0.005, 0.05, n))),
        }, index=idx)
    import yfinance as yf
    out = {}
    for etq, tks in (("bolsa", ["^GSPC"]), ("oro", ["GC=F", "GLD"])):
        for tk in tks:
            try:
                df = yf.download(tk, period="max", interval="1mo",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                s = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if len(s) > 200:
                    out[etq] = s
                    break
            except Exception:
                continue
    if len(out) < 2:
        return None
    return pd.DataFrame(out).dropna()


def _cargar_etfs(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(57)
        n = 3000
        idx = pd.bdate_range("2014-01-01", periods=n)
        return pd.DataFrame({tk: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
                             for tk in ("SPY", "GLD", "TLT", "XLU", "XLP")}, index=idx)
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(20 * 365.25))).isoformat()
    tks = ["SPY", "GLD", "TLT", "XLU", "XLP"]
    try:
        df = yf.download(tks, start=inicio, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
    except Exception:
        return None
    out = {}
    for tk in tks:
        try:
            s = df[tk]["Close"].dropna()
            if len(s) > 1000:
                out[tk] = s
        except Exception:
            continue
    return pd.DataFrame(out).ffill() if len(out) >= 4 else None


def _boot(x, seed=61, n_boot=3000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    k = len(x)
    if k < 24:
        return None
    rng = np.random.default_rng(seed)
    ms = np.array([x[rng.integers(0, k, k)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(ms, [5, 95]); m = float(np.mean(x))
    p = float(np.mean(ms <= 0)) if m > 0 else float(np.mean(ms >= 0))
    return {"n": k, "m": round(m, 3), "ic": [round(float(lo), 3), round(float(hi), 3)],
            "p": round(min(1.0, 2 * p), 4)}


def _met(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 24:
        return None
    eq = np.cumprod(1 + r)
    return {"cagr": round(float((eq[-1] ** (12 / len(r)) - 1) * 100), 1),
            "dd": round(float((eq / np.maximum.accumulate(eq) - 1).min() * 100), 1),
            "sharpe": round(float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(12)), 2)}


def _fila(nombre, est, bh, razon, color):
    df = pd.DataFrame({"e": est, "b": bh}).dropna()
    if len(df) < 60:
        return None
    r = _boot((df["e"].values - df["b"].values) * 100)
    if r is None:
        return None
    me, mb = _met(df["e"].values), _met(df["b"].values)
    if not me or not mb:
        return None
    return {"nombre": nombre, "razon": razon, "color": color, "n_eventos": len(df),
            "puntos": [{"etiqueta": "exceso mensual", "valor": r["m"], "ic_lo": r["ic"][0],
                        "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": (f"CAGR {me['cagr']}% vs {mb['cagr']}% · caída {me['dd']}% vs {mb['dd']}% · "
                      f"Sharpe {me['sharpe']} vs {mb['sharpe']}")}


def evaluar_combinaciones(sintetico=False):
    bloques = []

    # ---------- A) Estacionalidad del oro con historia larga ----------
    largo = _cargar_largo(sintetico)
    anios_largo = 0
    if largo is not None and len(largo) > 240:
        men = largo.resample("ME").last() if largo.index.freq is None else largo
        ret = men.pct_change().dropna()
        anios_largo = round(len(ret) / 12)
        bolsa, oro = ret["bolsa"], ret["oro"]
        for etq, meses in VENTANAS_MES.items():
            enmes = np.isin(ret.index.month, meses)
            est = pd.Series(np.where(enmes, oro.values, bolsa.values), index=ret.index)
            est = est - (COSTE / 100.0) * pd.Series(
                np.abs(np.diff(np.concatenate([[0], enmes.astype(float)]))), index=ret.index)
            b = _fila(f"A · Rotar al oro en {etq} ({anios_largo} años)", est, bolsa,
                      (f"Rotar de bolsa a oro durante {etq} y volver el resto del año. Con "
                       f"{anios_largo} años de historia, no solo 20."), "#e8b23a")
            if b:
                bloques.append(b)

    # ---------- B) Combinaciones defensivas ----------
    px = _cargar_etfs(sintetico)
    if px is not None and "SPY" in px.columns:
        men = px.resample("ME").last()
        ret = men.pct_change()
        spy = ret["SPY"]
        ma = men["SPY"].rolling(10).mean()
        dentro = (men["SPY"].shift(1) > ma.shift(1))
        camb = dentro.astype(float).diff().abs().fillna(0) * (COSTE / 100.0)
        defens = ((ret["XLU"] + ret["XLP"]) / 2.0) if {"XLU", "XLP"} <= set(ret.columns) else None
        perm = ((ret["SPY"] + ret["GLD"] + ret["TLT"]) / 3.0) if {"GLD", "TLT"} <= set(ret.columns) else None

        combos = []
        if defens is not None and perm is not None:
            combos.append(("B1 · Defensivos + permanente (mitad y mitad)",
                           (spy.where(dentro, defens) * 0.5 + perm * 0.5) - camb,
                           "Combina rotación defensiva con cartera permanente al 50%."))
            combos.append(("B2 · Permanente con filtro de tendencia",
                           perm.where(dentro, (ret["GLD"] + ret["TLT"]) / 2.0) - camb,
                           "Cartera permanente que sale de la bolsa cuando la tendencia se rompe."))
        if defens is not None and "GLD" in ret.columns:
            combos.append(("B3 · Defensivos + oro al romperse la tendencia",
                           spy.where(dentro, (defens + ret["GLD"]) / 2.0) - camb,
                           "Al romperse la tendencia, mitad defensivos y mitad oro."))
        if perm is not None:
            combos.append(("B4 · Permanente con más peso en bolsa (50/25/25)",
                           (ret["SPY"] * 0.5 + ret["GLD"] * 0.25 + ret["TLT"] * 0.25) - COSTE / 100.0,
                           "Cartera permanente menos conservadora, con la mitad en bolsa."))
        for nombre, est, razon in combos:
            b = _fila(nombre, est, spy, razon, "#5fb7c4")
            if b:
                bloques.append(b)

    if not bloques:
        return None

    pvals = [b["puntos"][0]["p"] for b in bloques]
    mask = figuras._bh(pvals, q=0.10)
    for b, ok in zip(bloques, mask):
        p = b["puntos"][0]
        p["sig_cruda"] = bool(p["ic_lo"] > 0 or p["ic_hi"] < 0)
        p["sig_fdr"] = bool(ok)
    n_fdr = int(np.sum(mask))

    detalle = "".join(
        f"<div class='vf-row'><span class='dot' style='background:{b['color']}'></span>"
        f"<b>{b['nombre']}</b></div><div class='ch-sub' style='margin:4px 0 12px 18px'>"
        f"{b['razon']}<br><i>{b['extra']}</i></div>" for b in bloques)
    for b in bloques:
        b["tipo"] = b["nombre"]
        b["nombre"] = f"{b['nombre']} · {b['n_eventos']} meses"

    return {
        "id": "combinaciones",
        "etiqueta": "Combinaciones y estacionalidad larga",
        "tipo": f"{len(bloques)} variantes probadas · FDR conjunto · vs comprar y mantener",
        "modelo": "combi",
        "figuras_panel": True,
        "intro": ("Dos frentes: la estacionalidad del oro con toda la historia disponible (la prueba "
                  "que le faltaba), y combinaciones de lo único que ha aportado en el laboratorio "
                  "—diversificación y rotación defensiva—. «Ventaja» positiva = bate a comprar y "
                  "mantener." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"<b>Se han probado {len(pvals)} variantes a la vez.</b> Probar muchas "
                     f"combinaciones garantiza encontrar «ganadoras» por azar, así que la corrección "
                     f"de Benjamini-Hochberg se aplica al conjunto: {n_fdr} sobreviven. Una variante "
                     f"que destaque sin corregir NO vale nada — es la trampa que este laboratorio "
                     f"lleva demostrando una y otra vez."),
        "nota": ("Recuerda mirar CAGR, caída máxima y Sharpe además del exceso: en todo este "
                 "laboratorio, lo que ha aportado valor no ha sido predecir sino gestionar riesgo, "
                 "y eso no se ve en el exceso de retorno. No es recomendación de inversión."),
    }
