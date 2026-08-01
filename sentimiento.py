"""
sentimiento.py — Sentimiento extremo del mercado como señal CONTRARIA (VIX).
===========================================================================
Hipótesis formulada ANTES de mirar los datos (opinión contraria, Fosback):

  - Tras un MIEDO extremo  (VIX en su decil más ALTO de los últimos ~2 años),
    el S&P 500 tiende a REBOTAR en las semanas siguientes.
  - Tras una EUFORIA extrema (VIX en su decil más BAJO), tiende a CORREGIR.

Es UNA hipótesis con fundamento económico, no una expedición de pesca: acotamos
el espacio a una idea defendible, la medimos fuera de muestra como exceso sobre la
tasa base, corregimos por multiple-testing (Benjamini-Hochberg) y aceptamos el
veredicto salga como salga. NO es asesoramiento financiero.

«Ventaja» positiva = la hipótesis se cumple (el mercado sube tras miedo / baja tras
euforia), por encima de lo que hace de media.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import data
import figuras                       # reutilizamos _boot_media y _bh (misma estadística)
from implied_vol import _cargar_indice

HZ = [5, 10, 21, 42, 63]
LAB = {5: "1 sem", 10: "2 sem", 21: "1 mes", 42: "2 meses", 63: "3 meses"}
W = 504          # ventana de referencia para los deciles (~2 años)
Q_ALTO = 0.90    # miedo extremo: VIX por encima del decil 90
Q_BAJO = 0.10    # euforia extrema: VIX por debajo del decil 10
GAP = 10         # separación mínima entre eventos (evita racimos)

SENALES = {
    "miedo":   (+1, "Miedo extremo (VIX alto) → rebote", "#6ec08a"),
    "euforia": (-1, "Euforia extrema (VIX bajo) → corrección", "#d2566a"),
}


def _sinteticos():
    rng = np.random.default_rng(11)
    n = 3000
    r = rng.normal(0.0003, 0.011, n)
    sp = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.date_range("2010-01-01", periods=n))
    # VIX sintético: sube cuando el mercado cae (miedo), con algo de ruido
    vol20 = pd.Series(r).rolling(20).std().bfill().values * np.sqrt(252) * 100
    vix = pd.Series(np.clip(vol20 + rng.normal(0, 3, n) + 8, 6, 80), index=sp.index)
    return vix, sp


def _cargar():
    sp = data.cargar_panel(["sp500"])["sp500"]
    vix = _cargar_indice("^VIX")
    return vix, sp


def _prueba_continua(vv, p, base, W, hz=(21, 63)):
    """Prueba continua (TODOS los días, mucha más potencia que los ~45 eventos):
    ¿predice el percentil del VIX el retorno futuro en exceso? Correlación con
    p-valor por bootstrap de bloque. Positiva = la hipótesis contraria se cumple."""
    v = pd.Series(vv)
    rank = v.rolling(W).apply(lambda a: float((a[:-1] < a[-1]).mean()), raw=True).values
    n = len(vv)
    out = {}
    for h in hz:
        fe, rk = [], []
        for t in range(W, n - h):
            if np.isfinite(rank[t]):
                fe.append(p[t + h] / p[t] - 1.0 - base[h]); rk.append(rank[t])
        fe = np.asarray(fe); rk = np.asarray(rk)
        if len(fe) < 200:
            continue
        corr = float(np.corrcoef(rk, fe)[0, 1])
        rng = np.random.default_rng(5); m = len(fe); bloque = h
        nb = int(np.ceil(m / bloque)); bidx = np.arange(bloque)
        cs = np.empty(1500)
        for i in range(1500):
            idx = (rng.integers(0, m, size=nb)[:, None] + bidx[None, :]).ravel() % m
            cs[i] = np.corrcoef(rk[idx], fe[idx])[0, 1]
        p1 = float(np.mean(cs <= 0)) if corr > 0 else float(np.mean(cs >= 0))
        out[h] = (round(corr, 3), round(min(1.0, 2 * p1), 4), len(fe))
    return out


def backtest_sentimiento(sintetico=False):
    vix, sp = _sinteticos() if sintetico else _cargar()
    if vix is None or sp is None:
        return None
    df = pd.DataFrame({"sp": sp, "vix": vix}).dropna()
    if len(df) < W + 300:
        return None
    fechas = [d.strftime("%Y-%m-%d") for d in df.index]
    p = df["sp"].values
    v = pd.Series(df["vix"].values)
    n = len(df)

    hi = v.rolling(W).quantile(Q_ALTO).values
    lo = v.rolling(W).quantile(Q_BAJO).values
    base = {h: float(np.nanmean(p[h:] / p[:-h] - 1.0)) for h in HZ}

    acc = {k: {h: [] for h in HZ} for k in SENALES}
    n_ev = {k: 0 for k in SENALES}
    last = {k: -10 ** 9 for k in SENALES}
    vv = v.values
    for t in range(W, n):
        if not (np.isfinite(hi[t]) and np.isfinite(lo[t]) and np.isfinite(hi[t - 1])):
            continue
        # MIEDO: el VIX entra en su decil alto
        if vv[t] > hi[t] and vv[t - 1] <= hi[t - 1] and (t - last["miedo"]) >= GAP:
            n_ev["miedo"] += 1; last["miedo"] = t
            for h in HZ:
                if t + h < n:
                    acc["miedo"][h].append((+1) * (p[t + h] / p[t] - 1.0 - base[h]) * 100.0)
        # EUFORIA: el VIX entra en su decil bajo
        if vv[t] < lo[t] and vv[t - 1] >= lo[t - 1] and (t - last["euforia"]) >= GAP:
            n_ev["euforia"] += 1; last["euforia"] = t
            for h in HZ:
                if t + h < n:
                    acc["euforia"][h].append((-1) * (p[t + h] / p[t] - 1.0 - base[h]) * 100.0)

    celdas = []
    for k in SENALES:
        for h in HZ:
            x = acc[k][h]
            if len(x) < 15:
                continue
            m, ic, p1 = figuras._boot_media(x, bloque=max(10, h // 2))
            celdas.append({"k": k, "h": h, "valor": round(m, 2), "ic_lo": round(ic[0], 2),
                           "ic_hi": round(ic[1], 2), "n": len(x), "p": min(1.0, 2 * p1)})
    if not celdas:
        return None
    mask = figuras._bh([c["p"] for c in celdas], q=0.10)
    for c, ok in zip(celdas, mask):
        c["sig_cruda"] = bool(c["ic_lo"] > 0 or c["ic_hi"] < 0)
        c["sig_fdr"] = bool(ok)

    figs = []
    for k, (_d, nombre, color) in SENALES.items():
        pts = [c for c in celdas if c["k"] == k]
        if not pts:
            continue
        figs.append({"tipo": k, "nombre": nombre, "color": color, "n_eventos": n_ev[k],
                     "puntos": [{"etiqueta": LAB[c["h"]], "valor": c["valor"], "ic_lo": c["ic_lo"],
                                 "ic_hi": c["ic_hi"], "n": c["n"], "sig_cruda": c["sig_cruda"],
                                 "sig_fdr": c["sig_fdr"]} for c in pts]})
    if not figs:
        return None
    n_fdr = sum(1 for c in celdas if c["sig_fdr"])

    # Prueba continua de más potencia (todos los días, no solo los ~45 extremos)
    cont = _prueba_continua(vv, p, base, W)
    cont_txt = ""
    if cont:
        partes = []
        etiq = {21: "1 mes", 63: "3 meses"}
        for h, (c, pp, _nn) in cont.items():
            partes.append(f"a {etiq.get(h, str(h))}: correlación {c:+.2f} (p={pp})")
        nobs = list(cont.values())[0][2]
        cont_txt = (f" || Prueba continua (todos los días, {nobs} obs, mucha más potencia que los "
                    f"eventos): ¿predice el nivel del VIX el retorno futuro? " + "; ".join(partes)
                    + ". Correlación positiva = la hipótesis contraria se cumple.")
    return {
        "id": "sentimiento_vix",
        "etiqueta": "Sentimiento extremo (VIX)",
        "tipo": "Opinión contraria · event study · S&P 500",
        "modelo": "sentimiento",
        "figuras_panel": True,
        "intro": ("Hipótesis contraria formulada de antemano: tras MIEDO extremo (VIX en su decil "
                  "más alto de los últimos ~2 años) el mercado tiende a rebotar; tras EUFORIA extrema "
                  "(decil más bajo), a corregir. «Ventaja» positiva = la hipótesis se cumple, por "
                  "encima de la tasa base del propio índice."),
        "figuras": figs,
        "n_celdas": len(celdas),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se evalúan {len(celdas)} combinaciones señal×horizonte. Tras corregir por "
                     f"multiple-testing (Benjamini-Hochberg, FDR 10%), {n_fdr} sobreviven. La columna "
                     f"«Tras FDR» es la única que cuenta; la «cruda» es la trampa." + cont_txt),
        "nota": ("Es UNA hipótesis con fundamento (opinión contraria, Fosback), no una búsqueda a "
                 "ciegas. Aceptamos el veredicto sea cual sea. Ojo: aunque el efecto exista, suele ser "
                 "débil y comerse los costes. No es recomendación de inversión."),
    }


def estado_actual(sintetico=False):
    """¿Está el VIX en un extremo hoy? Devuelve dict con la señal contraria o None."""
    vix, sp = _sinteticos() if sintetico else _cargar()
    if vix is None:
        return None
    v = pd.Series(vix).dropna()
    if len(v) < W + 5:
        return None
    hi = float(v.rolling(W).quantile(Q_ALTO).iloc[-1])
    lo = float(v.rolling(W).quantile(Q_BAJO).iloc[-1])
    actual = float(v.iloc[-1])
    pct = float((v.iloc[-W:] < actual).mean() * 100)
    if actual > hi:
        return {"senal": "miedo", "vix": round(actual, 1), "percentil": round(pct),
                "lectura": "miedo extremo → sesgo contrario al alza"}
    if actual < lo:
        return {"senal": "euforia", "vix": round(actual, 1), "percentil": round(pct),
                "lectura": "euforia extrema → sesgo contrario a la baja"}
    return {"senal": None, "vix": round(actual, 1), "percentil": round(pct),
            "lectura": "sentimiento en zona normal"}
