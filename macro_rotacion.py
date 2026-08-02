"""
macro_rotacion.py — Seis estrategias de rotación y defensa, testeadas juntas.
============================================================================
Cada una con su razón económica, formulada antes de mirar. Todas se comparan
contra comprar y mantener el S&P 500, con coste por rotación, y se corrigen
CONJUNTAMENTE por multiple-testing.

  R1. ROTACIÓN ESTACIONAL S&P ↔ ORO. Agosto-septiembre son los peores meses de la
      bolsa y de los mejores del oro. En vez de salirse (que cuesta rentabilidad),
      rotar al oro en esos meses y volver después.

  R2. CARTERA PERMANENTE. Repartir entre bolsa, oro y bonos con rebalanceo anual.
      Razón: cada activo funciona en un régimen distinto (crecimiento, inflación,
      deflación), así que la cartera aguanta cualquier escenario.

  R3. CURVA DE TIPOS. Salir de bolsa cuando la curva 10a-3m está invertida (proxy
      con ETFs de bonos: TLT/SHY). Es la señal de recesión mejor documentada.

  R4. DIFERENCIAL DE CRÉDITO. Salir cuando los bonos de alto rendimiento (HYG) se
      deterioran frente a los de calidad (LQD). Razón: el crédito anticipa el
      estrés antes que la bolsa; es un proxy público de liquidez.

  R5. FILTRO DEL DÓLAR. Reducir exposición cuando el dólar se fortalece. Razón:
      dólar fuerte drena liquidez global y perjudica a los activos de riesgo.

  R6. ROTACIÓN DEFENSIVA. Cuando la tendencia se rompe, ir a sectores defensivos
      (utilities, consumo básico) en vez de salir a liquidez.

Todo con rebalanceo mensual y coste aplicado. NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
COSTE = 0.05          # % por cambio de posición
TICKERS = ["SPY", "GLD", "TLT", "SHY", "HYG", "LQD", "UUP", "XLU", "XLP"]


def _cargar(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(41)
        n = 3200
        idx = pd.bdate_range("2013-01-01", periods=n)
        out = {}
        for tk in TICKERS:
            out[tk] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
        return pd.DataFrame(out, index=idx)
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    try:
        df = yf.download(TICKERS, start=inicio, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
    except Exception:
        return None
    cierres = {}
    for tk in TICKERS:
        try:
            s = df[tk]["Close"].dropna()
            if len(s) > 1000:
                cierres[tk] = s
        except Exception:
            continue
    if "SPY" not in cierres:
        return None
    return pd.DataFrame(cierres).ffill()


def _boot(x, seed=37, n_boot=3000):
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


def _resultado(nombre, razon, color, est, bh, extra=""):
    df = pd.DataFrame({"est": est, "bh": bh}).dropna()
    if len(df) < 48:
        return None
    e, b = df["est"].values, df["bh"].values
    r = _boot((e - b) * 100)
    if r is None:
        return None
    me, mb = _met(e), _met(b)
    if not me or not mb:
        return None
    return {"nombre": nombre, "razon": razon, "color": color, "n_eventos": len(df),
            "puntos": [{"etiqueta": "exceso mensual", "valor": r["m"], "ic_lo": r["ic"][0],
                        "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": (f"CAGR {me['cagr']}% vs {mb['cagr']}% · caída máx {me['dd']}% vs {mb['dd']}% · "
                      f"Sharpe {me['sharpe']} vs {mb['sharpe']}. {extra}")}


def evaluar_macro(sintetico=False):
    px = _cargar(sintetico)
    if px is None:
        return None
    men = px.resample("ME").last()
    ret = men.pct_change()
    if "SPY" not in ret.columns:
        return None
    spy = ret["SPY"]
    bloques = []

    # R1: rotación estacional S&P <-> oro (agosto y septiembre en oro)
    if "GLD" in ret.columns:
        mes = ret.index.month
        est = pd.Series(np.where(np.isin(mes, [8, 9]), ret.get("GLD", spy), spy),
                        index=ret.index) - (2 * COSTE / 100.0) * np.isin(mes, [8, 9, 10])
        bloques.append(_resultado(
            "R1 · Rotación estacional S&P ↔ oro (ago-sep)",
            ("Agosto y septiembre son los peores meses de la bolsa y de los mejores del oro. "
             "En vez de salirse (que cuesta rentabilidad), rotar al oro esos dos meses."),
            "#e8b23a", est, spy))

    # R2: cartera permanente (bolsa/oro/bonos), rebalanceo mensual
    if {"GLD", "TLT"} <= set(ret.columns):
        est = (ret["SPY"] + ret["GLD"] + ret["TLT"]) / 3.0 - COSTE / 100.0
        bloques.append(_resultado(
            "R2 · Cartera permanente (bolsa + oro + bonos)",
            ("Cada activo funciona en un régimen distinto (crecimiento, inflación, deflación), "
             "así que la cartera debería aguantar cualquier escenario con menos sobresaltos."),
            "#88c0d0", est, spy))

    # R3: curva de tipos (proxy TLT vs SHY: si el largo va peor que el corto de forma
    # sostenida, la curva se aplana/invierte -> señal de recesión)
    if {"TLT", "SHY"} <= set(ret.columns):
        señal = (men["TLT"] / men["TLT"].shift(12) - men["SHY"] / men["SHY"].shift(12))
        dentro = (señal.shift(1) > 0)
        camb = dentro.astype(float).diff().abs().fillna(0)
        est = spy.where(dentro, 0.0) - camb * (COSTE / 100.0)
        bloques.append(_resultado(
            "R3 · Filtro de curva de tipos (bonos largos vs cortos)",
            ("La inversión de la curva es la señal de recesión mejor documentada. Proxy con ETFs: "
             "si los bonos largos rinden peor que los cortos, hay tensión."),
            "#6ec08a", est, spy))

    # R4: diferencial de crédito (HYG vs LQD)
    if {"HYG", "LQD"} <= set(ret.columns):
        señal = (men["HYG"] / men["HYG"].shift(6) - men["LQD"] / men["LQD"].shift(6))
        dentro = (señal.shift(1) > 0)
        camb = dentro.astype(float).diff().abs().fillna(0)
        est = spy.where(dentro, 0.0) - camb * (COSTE / 100.0)
        bloques.append(_resultado(
            "R4 · Filtro de crédito (alto rendimiento vs calidad)",
            ("El crédito anticipa el estrés antes que la bolsa: cuando los bonos basura se "
             "deterioran frente a los de calidad, hay tensión de liquidez. Proxy público del "
             "índice de liquidez que no es accesible."),
            "#d2566a", est, spy))

    # R5: filtro del dólar
    if "UUP" in ret.columns:
        señal = (men["UUP"] / men["UUP"].shift(6) - 1.0)
        dentro = (señal.shift(1) < 0.02)          # fuera si el dólar se fortalece con fuerza
        camb = dentro.astype(float).diff().abs().fillna(0)
        est = spy.where(dentro, 0.0) - camb * (COSTE / 100.0)
        bloques.append(_resultado(
            "R5 · Filtro del dólar",
            ("Un dólar fuerte drena liquidez global y perjudica a los activos de riesgo. "
             "Se reduce exposición cuando el dólar se aprecia con fuerza."),
            "#b48ad6", est, spy))

    # R6: rotación defensiva (a utilities/consumo básico en vez de salir)
    if {"XLU", "XLP"} <= set(ret.columns):
        ma = men["SPY"].rolling(10).mean()
        dentro = (men["SPY"].shift(1) > ma.shift(1))
        defensivo = (ret["XLU"] + ret["XLP"]) / 2.0
        camb = dentro.astype(float).diff().abs().fillna(0)
        est = spy.where(dentro, defensivo) - camb * (COSTE / 100.0)
        bloques.append(_resultado(
            "R6 · Rotación defensiva (utilities y consumo básico)",
            ("Cuando la tendencia se rompe, en vez de salir a liquidez (que cuesta rentabilidad) "
             "rotar a sectores defensivos, que caen menos pero siguen dando algo."),
            "#5fb7c4", est, spy))

    bloques = [b for b in bloques if b]
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
        "id": "macro_rotacion",
        "etiqueta": "Rotación y filtros macro",
        "tipo": f"{len(bloques)} estrategias · FDR conjunto · neto de costes · vs comprar y mantener",
        "modelo": "macro",
        "figuras_panel": True,
        "intro": ("Seis formas de rotar o defenderse en vez de simplemente estar dentro: "
                  "estacionalidad, diversificación, curva de tipos, crédito, dólar y sectores "
                  "defensivos. Todas comparadas contra comprar y mantener el S&P 500, con coste. "
                  "«Ventaja» positiva = bate a estar quieto." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se prueban {len(pvals)} estrategias A LA VEZ, así que la corrección de "
                     f"Benjamini-Hochberg se aplica al conjunto: {n_fdr} sobreviven. Ojo: el exceso "
                     f"de retorno no lo es todo — mira también CAGR y caída máxima de cada una, "
                     f"porque una estrategia puede rendir algo menos y aun así ser preferible si "
                     f"reduce mucho el riesgo."),
        "nota": ("Lo que NO se ha podido testear y conviene decir: el índice de liquidez global de "
                 "Howell es propietario; las primas de riesgo en tiempo real requieren datos de "
                 "beneficios que no tenemos; y «predecir crisis» en abstracto ya ha fallado en todas "
                 "las formas probadas en este laboratorio. Aquí se usan proxies públicos (crédito, "
                 "curva, dólar) que son lo más cercano accesible. No es recomendación de inversión."),
    }
