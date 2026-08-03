"""
mineras.py — Lote M: la tesis macro de las mineras de oro, medida a nivel índice.
=================================================================================
Tesis del método a examen: las mineras son apalancamiento operativo sobre el oro
(coste de extracción ~fijo → el margen amplifica el movimiento del metal), el
petróleo es su coste principal, y la selección por salud financiera decide quién
cobra ese apalancamiento y quién diluye. La SELECCIÓN no se puede backtestear
honestamente (sin fundamentales point-in-time y con el peor sesgo de
supervivencia que existe: las quebradas ya no cotizan) — esa parte va al
forward-test (forward_mineras.py). Aquí se mide lo que SÍ se puede medir limpio:
la parte macro, solo con precios de índices.

CONVENCIÓN: efectivo al tipo corto (^IRX), coste 0,05% por cambio de posición.
Benchmark de las celdas de señal: mantener GDX (o la mezcla 50/50), no el
efectivo — la vara es "¿la señal añade algo sobre tener el sector?".

SEIS CELDAS, RAZÓN ECONÓMICA ESCRITA ANTES DE MIRAR DATOS:

  M1. GDX vs oro (GLD), comprar y mantener. ¿Paga el apalancamiento operativo
      por sí solo? Predicción: NO — dilución, sobrecostes y mala asignación de
      capital se comen el apalancamiento; es el hallazgo clásico del sector.
      Esta celda es la base de todo: si M1 fuera positivo, no haría falta
      seleccionar nada.
  M2. GDXJ vs GDX, comprar y mantener. ¿Pagan las juniors su riesgo extra?
      Predicción: NO — beta más alta sin prima (misma lección que DEFβ:
      la lotería se sobrepaga).
  M3. Valor relativo: cuando el ratio GDX/oro está barato (z<-1 sobre 252
      sesiones), mineras; si no, oro. Vs 50/50 estático. Razón: si mineras y
      oro comparten fundamento, el ratio debería revertir; comprar el lado
      barato cosecha la reversión.
  M4. Margen: proxy = oro/petróleo. Si el margen sube (variación 3 meses
      positiva), mineras; si no, oro. Vs 50/50 estático. Razón: el beneficio
      minero es precio menos coste energético; el mercado podría infrarreaccionar
      a la expansión de márgenes.
  M5. Tendencia: oro sobre su media de 10 meses → GDX; si no → efectivo.
      Vs GDX comprar y mantener. Razón: las mineras son apalancamiento sobre
      el oro, y el apalancamiento solo se quiere con el subyacente en tendencia;
      fuera de tendencia, el apalancamiento amplifica el castigo.
  M6. Síntesis pre-registrada (margen Y tendencia → GDX; solo tendencia → oro;
      nada → efectivo). Vs GDX comprar y mantener. Es la versión medible del
      método completo sin la capa de selección.

PREDICCIONES: M1 y M2 negativas (y eso VALIDA la necesidad de seleccionar, que
es justo lo que defiende el método del usuario del laboratorio); M3-M4 ruido
probable; M5 es la de más probabilidad a priori (reduce el desastre en los
ciclos bajistas del oro, que son largos); M6 vive o muere con M5.

CONTROL NEGATIVO: M5 con la señal permutada. FDR conjunto sobre las 6 celdas.
Contador del laboratorio: ~95 → ~101. NO es asesoramiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
COSTE = 0.05
RF_FALLBACK = 0.03
N_CELDAS_DECLARADAS = 6
TICKERS = ["GDX", "GDXJ", "GLD"]
PETROLEO = ["CL=F", "USO"]          # futuro delantero; USO de repuesto


# ---------------------------------------------------------------- datos ----
def _cargar(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(53)
        n = 3200
        idx = pd.bdate_range("2013-01-01", periods=n)
        oro = np.cumsum(rng.normal(0.0002, 0.009, n))
        out = {"GLD": 100 * np.exp(oro),
               "GDX": 100 * np.exp(1.8 * oro + np.cumsum(rng.normal(-0.0001, 0.012, n))),
               "GDXJ": 100 * np.exp(2.2 * oro + np.cumsum(rng.normal(-0.0002, 0.016, n))),
               "CL=F": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, n)))}
        px = pd.DataFrame(out, index=idx)
        rf = pd.Series(RF_FALLBACK, index=px.resample("ME").last().index)
        return px, rf, "CL=F"
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    out = {}
    try:
        df = yf.download(TICKERS, start=inicio, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
        for tk in TICKERS:
            try:
                s = df[tk]["Close"].dropna()
                if len(s) > 1000:
                    out[tk] = s
            except Exception:
                continue
    except Exception:
        return None, None, None
    oil_col = None
    for tk in PETROLEO:
        try:
            s = yf.download(tk, start=inicio, auto_adjust=True,
                            progress=False)["Close"].dropna().squeeze()
            if len(s) > 1000:
                out[tk] = s; oil_col = tk
                break
        except Exception:
            continue
    if "GDX" not in out or "GLD" not in out:
        return None, None, None
    px = pd.DataFrame(out).sort_index().ffill()
    rf = None
    try:
        irx = yf.download("^IRX", start=inicio, auto_adjust=True,
                          progress=False)["Close"].dropna()
        if len(irx) > 500:
            rf = (irx / 100.0).resample("ME").last().reindex(
                px.resample("ME").last().index).ffill().squeeze()
    except Exception:
        rf = None
    if rf is None:
        rf = pd.Series(RF_FALLBACK, index=px.resample("ME").last().index)
        rf.attrs["fallback"] = True
    return px, rf, oil_col


# ----------------------------------------------------------- estadística ----
def _boot(x, seed=97, n_boot=3000):
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


def _fila(nombre, razon, color, est, bh, bench_txt, seed=97):
    df = pd.DataFrame({"e": est, "b": bh}).dropna()
    if len(df) < 60:
        return None
    r = _boot((df["e"].values - df["b"].values) * 100, seed=seed)
    if r is None:
        return None
    me, mb = _met(df["e"].values), _met(df["b"].values)
    if not me or not mb:
        return None
    return {"nombre": nombre, "razon": razon, "color": color, "n_eventos": len(df),
            "puntos": [{"etiqueta": f"exceso mensual vs {bench_txt}", "valor": r["m"],
                        "ic_lo": r["ic"][0], "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": (f"CAGR {me['cagr']}% vs {mb['cagr']}% · caída {me['dd']}% vs {mb['dd']}% · "
                      f"<b>Sharpe {me['sharpe']} vs {mb['sharpe']}</b> (benchmark: {bench_txt}).")}


def _conmutada(senal, alto, bajo, rf_men=None, coste=COSTE):
    """Retorno mensual de conmutar entre dos series según señal booleana (con
    información del mes anterior). `bajo` puede ser None → efectivo."""
    senal = senal.astype(float)
    if bajo is None:
        if rf_men is None:
            raise ValueError("falta rf")
        bajo = rf_men.ffill() / 12.0
    r = senal * alto + (1.0 - senal) * bajo
    camb = senal.diff().abs().fillna(0)
    return (r - camb * (coste / 100.0)).dropna()


# --------------------------------------------------------------- lote ----
def evaluar(sintetico=False):
    px, rf, oil_col = _cargar(sintetico)
    if px is None:
        return None
    men = px.resample("ME").last()
    ret = men.pct_change().dropna(how="all")
    if "GDX" not in ret.columns or "GLD" not in ret.columns or len(ret) < 80:
        return None
    gdx, oro = ret["GDX"], ret["GLD"]
    rf_fb = bool(getattr(rf, "attrs", {}).get("fallback", False))
    bloques, faltan = [], []
    seed = 400

    # M1 · Apalancamiento operativo crudo
    b = _fila("M1 · GDX vs oro, comprar y mantener",
              ("¿Paga el apalancamiento operativo por sí solo? Predicción: NO — dilución, "
               "sobrecostes y mala asignación de capital se lo comen; es el hallazgo clásico del "
               "sector. Si M1 fuera positivo, no haría falta seleccionar nada; si es negativo, "
               "valida la premisa del método: el índice es la trampa, la selección es la tesis."),
              "#e8b23a", gdx, oro.reindex(gdx.index), "oro (GLD)", seed)
    if b: bloques.append(b)
    seed += 1

    # M2 · Juniors
    if "GDXJ" in ret.columns:
        b = _fila("M2 · GDXJ vs GDX, comprar y mantener",
                  ("¿Pagan las juniors su riesgo extra? Predicción: NO — beta más alta sin prima; "
                   "la lotería se sobrepaga (misma lección estructural que la beta baja del lote "
                   "V3, en espejo)."),
                  "#d4736a", ret["GDXJ"].dropna(),
                  gdx.reindex(ret["GDXJ"].dropna().index), "GDX", seed)
        if b: bloques.append(b)
    else:
        faltan.append("GDXJ")
    seed += 1

    # Señales (todas con información del mes anterior: shift(1))
    ratio = (men["GDX"] / men["GLD"])
    z = ((ratio - ratio.rolling(12).mean()) / ratio.rolling(12).std()).shift(1)
    mezcla5050 = (gdx * 0.5 + oro * 0.5).dropna()

    # M3 · Valor relativo
    s3 = (z < -1.0)
    est3 = _conmutada(s3.reindex(gdx.index), gdx, oro.reindex(gdx.index))
    b = _fila("M3 · Mineras cuando están baratas contra el oro (z del ratio < -1)",
              ("Si mineras y oro comparten fundamento, su ratio debería revertir: comprar el lado "
               "barato cosecha la reversión. Señal con datos del mes anterior; el resto del "
               "tiempo, oro. Benchmark: 50/50 estático."),
              "#5fb7c4", est3, mezcla5050.reindex(est3.index), "50/50 GDX-oro", seed)
    if b: bloques.append(b)
    seed += 1

    # M4 · Margen (oro/petróleo)
    if oil_col and oil_col in men.columns:
        margen = (men["GLD"] / men[oil_col])
        s4 = (margen.pct_change(3) > 0).shift(1)
        est4 = _conmutada(s4.reindex(gdx.index).fillna(False), gdx, oro.reindex(gdx.index))
        b = _fila("M4 · Mineras cuando el margen (oro/petróleo) se expande",
                  ("El beneficio minero es precio del metal menos coste energético: si el ratio "
                   "oro/petróleo sube en 3 meses, los márgenes se expanden y el mercado podría "
                   "infrarreaccionar. Benchmark: 50/50 estático."),
                  "#8fbf9f", est4, mezcla5050.reindex(est4.index), "50/50 GDX-oro", seed)
        if b: bloques.append(b)
    else:
        faltan.append("petróleo (CL=F/USO)")
    seed += 1

    # M5 · Tendencia del oro
    ma10 = men["GLD"].rolling(10).mean()
    s5 = (men["GLD"] > ma10).shift(1)
    est5 = _conmutada(s5.reindex(gdx.index).fillna(False), gdx, None, rf_men=rf.reindex(gdx.index))
    b = _fila("M5 · GDX solo con el oro sobre su media de 10 meses",
              ("Las mineras son apalancamiento sobre el oro, y el apalancamiento solo se quiere "
               "con el subyacente en tendencia: fuera de ella amplifica el castigo, y los ciclos "
               "bajistas del oro duran años. Si no hay tendencia: efectivo al tipo corto. "
               "Benchmark: GDX comprar y mantener. La celda de más probabilidad a priori."),
              "#b48ad6", est5, gdx.reindex(est5.index), "GDX", seed)
    if b: bloques.append(b)
    seed += 1

    # M6 · Síntesis
    if oil_col and oil_col in men.columns:
        s_marg = (margen.pct_change(3) > 0).shift(1).reindex(gdx.index).fillna(False)
        s_tend = s5.reindex(gdx.index).fillna(False)
        rf_m = rf.reindex(gdx.index).ffill() / 12.0
        pos = pd.Series(0.0, index=gdx.index)
        r6 = pd.Series(np.where(s_tend & s_marg, gdx,
                                np.where(s_tend, oro.reindex(gdx.index), rf_m)),
                       index=gdx.index)
        estado = (s_tend.astype(int) + (s_tend & s_marg).astype(int))
        camb = estado.diff().abs().fillna(0).clip(0, 1)
        est6 = (r6 - camb * (COSTE / 100.0)).dropna()
        b = _fila("M6 · Síntesis: margen y tendencia (mineras / oro / efectivo)",
                  ("La versión medible del método completo, sin la capa de selección: mineras solo "
                   "con margen expandiéndose Y oro en tendencia; solo tendencia → oro; nada → "
                   "efectivo. Benchmark: GDX comprar y mantener. Vive o muere con M5."),
                  "#7a9fd4", est6, gdx.reindex(est6.index), "GDX", seed)
        if b: bloques.append(b)
    seed += 1

    if not bloques:
        return None

    # Control negativo: M5 con señal permutada
    ctrl_txt = ""
    try:
        rng = np.random.default_rng(314)
        s_perm = pd.Series(rng.permutation(s5.reindex(gdx.index).fillna(False).values),
                           index=gdx.index)
        est_c = _conmutada(s_perm, gdx, None, rf_men=rf.reindex(gdx.index))
        dfc = pd.DataFrame({"e": est_c, "b": gdx}).dropna()
        rc = _boot((dfc["e"].values - dfc["b"].values) * 100, seed=999)
        if rc:
            ctrl_txt = (f"<br><b>Control negativo:</b> M5 con la señal de tendencia permutada da "
                        f"{rc['m']:+.3f}% mensual frente a GDX (p={rc['p']}). Se espera ruido.")
    except Exception:
        ctrl_txt = ""

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

    aviso = ""
    if faltan:
        aviso = ("<br><b>Datos no disponibles en esta corrida:</b> " + ", ".join(faltan) +
                 ". Las celdas afectadas no se evalúan.")
    if rf_fb:
        aviso += "<br><b>Aviso:</b> sin ^IRX; tipo corto constante del 3% anual."

    return {
        "id": "mineras_macro",
        "etiqueta": "Lote M · Mineras de oro (tesis macro)",
        "tipo": f"{len(bloques)} de {N_CELDAS_DECLARADAS} celdas declaradas · FDR conjunto · neto de costes",
        "modelo": "volobj",
        "figuras_panel": True,
        "intro": (
            "La tesis de las mineras, partida en dos con honestidad: la SELECCIÓN por salud "
            "financiera no se puede backtestear limpio (sin fundamentales point-in-time y con el "
            "peor sesgo de supervivencia posible: las quebradas ya no cotizan) y va al "
            "forward-test en vivo. Aquí se mide lo que sí se puede: la parte macro, a nivel "
            "índice, solo con precios. Predicciones previas: M1 y M2 negativas —lo que "
            "paradójicamente VALIDA la premisa del método: si el índice minero pierde contra el "
            "oro, tener el índice es la trampa y seleccionar es la única tesis defendible—; M5 "
            "es la de más probabilidad a priori; M6 vive o muere con ella." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Corrección conjunta sobre {len(pvals)} celdas (Benjamini-Hochberg, FDR 10%): "
                     f"{n_fdr} sobreviven. El laboratorio supera las ~100 pruebas acumuladas: nada "
                     f"de lo que brille aquí se cree sin pasar por el forward-test."),
        "nota": (ctrl_txt + aviso +
                 "<br>Este lote NO evalúa la selección de mineras concretas: eso solo puede "
                 "juzgarse hacia adelante y está en su propio registro en vivo. No es "
                 "recomendación de inversión."),
    }
