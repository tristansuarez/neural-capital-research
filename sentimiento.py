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


def _corr_boot(rk, fe, bloque, seed=5, nb=1500):
    if len(fe) < 100:
        return None
    corr = float(np.corrcoef(rk, fe)[0, 1])
    rng = np.random.default_rng(seed); m = len(fe)
    k = int(np.ceil(m / bloque)); bidx = np.arange(bloque)
    cs = np.empty(nb)
    for i in range(nb):
        idx = (rng.integers(0, m, size=k)[:, None] + bidx[None, :]).ravel() % m
        cs[i] = np.corrcoef(rk[idx], fe[idx])[0, 1]
    p1 = float(np.mean(cs <= 0)) if corr > 0 else float(np.mean(cs >= 0))
    return (round(corr, 3), round(min(1.0, 2 * p1), 4), len(fe))


def robustez_sentimiento(vv, p, base, fechas, W, h=63):
    """Somete el efecto VIX→retorno a torturas que un artefacto NO supera:
    sin crisis, fuera de muestra, monotonicidad y concentración."""
    v = pd.Series(vv)
    rank = v.rolling(W).apply(lambda a: float((a[:-1] < a[-1]).mean()), raw=True).values
    n = len(vv)
    fe, rk, ff = [], [], []
    for t in range(W, n - h):
        if np.isfinite(rank[t]):
            fe.append(p[t + h] / p[t] - 1.0 - base[h]); rk.append(rank[t]); ff.append(fechas[t])
    fe = np.asarray(fe); rk = np.asarray(rk); ff = np.asarray(ff)
    if len(fe) < 300:
        return None

    def _crisis(s):
        ym = s[:7]
        return ("2008-08" <= ym <= "2009-06") or ("2020-02" <= ym <= "2020-06") or ("2022-01" <= ym <= "2022-10")

    res = {}
    res["completo"] = _corr_boot(rk, fe, h)
    m = np.array([not _crisis(s) for s in ff])
    res["sin_crisis"] = _corr_boot(rk[m], fe[m], h)
    mid = len(fe) // 2
    res["primera"] = _corr_boot(rk[:mid], fe[:mid], h)
    res["segunda"] = _corr_boot(rk[mid:], fe[mid:], h)
    # concentración: quitar el 5% de días de más miedo
    thr = np.quantile(rk, 0.95)
    m2 = rk < thr
    res["sin_top5"] = _corr_boot(rk[m2], fe[m2], h)
    # monotonicidad: retorno medio futuro por quintil del percentil de VIX
    qs = np.quantile(rk, [0.2, 0.4, 0.6, 0.8])
    tramos = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], qs[3]), (qs[3], 2)]
    res["quintiles"] = [round(float(np.mean(fe[(rk > a) & (rk <= b)]) * 100), 2)
                        if ((rk > a) & (rk <= b)).sum() else None for a, b in tramos]
    return res


def episodios_sentimiento(vv, p, base, fechas, W, h=63, q=0.80, gap=10):
    """¿Cuántos episodios INDEPENDIENTES de miedo extremo forman el efecto? Agrupa
    los días del quintil alto en rachas (une huecos <= gap), toma UNA observación por
    episodio (entrada = primer día de miedo extremo) y hace bootstrap sobre episodios,
    que es la unidad realmente independiente. Es la prueba de tamaño muestral efectivo."""
    from collections import Counter
    v = pd.Series(vv)
    rank = v.rolling(W).apply(lambda a: float((a[:-1] < a[-1]).mean()), raw=True).values
    n = len(vv)
    alto = np.array([bool(np.isfinite(rank[t]) and rank[t] >= q) for t in range(n)])

    eps = []
    t = W
    while t < n - h:
        if alto[t]:
            ini = t; last = t; tt = t + 1
            while tt < n and (alto[tt] or (tt - last) <= gap):
                if alto[tt]:
                    last = tt
                tt += 1
            if ini + h < n:
                eps.append((fechas[ini], round((p[ini + h] / p[ini] - 1.0 - base[h]) * 100, 2)))
            t = last + 1
        else:
            t += 1

    if len(eps) < 5:
        return None
    fes = np.array([e[1] for e in eps]); k = len(fes); m = float(np.mean(fes))
    rng = np.random.default_rng(7); ms = np.empty(3000)
    for i in range(3000):
        ms[i] = fes[rng.integers(0, k, k)].mean()
    lo, hi = np.percentile(ms, [5, 95]); p1 = float(np.mean(ms <= 0))
    positivos = int(np.sum(fes > 0))
    anios = Counter(e[0][:4] for e in eps)
    return {"n": k, "media": round(m, 2), "ic": [round(float(lo), 2), round(float(hi), 2)],
            "p": round(min(1.0, 2 * p1), 4), "positivos": positivos,
            "anios": len(anios), "por_anio": dict(sorted(anios.items()))}


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

    # Torturas de robustez sobre el efecto a 3 meses
    rob = robustez_sentimiento(vv, p, base, fechas, W, h=63)
    rob_txt = ""
    if rob:
        def _c(x):
            return f"corr {x[0]:+.2f} (p={x[1]}, n={x[2]})" if x else "—"
        qs = rob.get("quintiles", [])
        rob_txt = (
            "<br><br><b>Pruebas de robustez (efecto a 3 meses — un artefacto no las supera):</b>"
            f"<br>• Completo: {_c(rob['completo'])}"
            f"<br>• <b>Sin crisis</b> (2008, 2020, 2022): {_c(rob['sin_crisis'])} — si aquí desaparece, eran solo las crisis"
            f"<br>• <b>Fuera de muestra</b>: 1ª mitad {_c(rob['primera'])} · 2ª mitad {_c(rob['segunda'])} — debe aguantar en la 2ª"
            f"<br>• <b>Sin el 5% de días de más miedo</b>: {_c(rob['sin_top5'])} — si se cae, lo mueven poquísimos días"
            f"<br>• <b>Monotonicidad</b> (retorno medio a 3m por quintil de VIX, de menos a más miedo, %): "
            f"{qs} — debería crecer de izquierda a derecha")

    # Episodios independientes: ¿cuántos momentos distintos forman el efecto de cola?
    ep = episodios_sentimiento(vv, p, base, fechas, W, h=63)
    if ep:
        rob_txt += (
            "<br><br><b>Episodios independientes (la prueba que decide la fiabilidad):</b>"
            f"<br>El efecto de cola lo forman <b>{ep['n']} episodios</b> distintos de miedo extremo, "
            f"repartidos en {ep['anios']} años; {ep['positivos']} de {ep['n']} rebotaron. "
            f"Retorno medio por episodio a 3m: <b>{ep['media']:+.2f}%</b>, IC90 {ep['ic']}, "
            f"p={ep['p']} (bootstrap sobre episodios, no sobre días). "
            f"Reparto por año: {ep['por_anio']}. "
            f"— Si son muchos episodios en muchos años y p sigue baja, es de fiar; si son 4-6 episodios, es frágil.")
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
                 "débil y comerse los costes. No es recomendación de inversión." + rob_txt),
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
