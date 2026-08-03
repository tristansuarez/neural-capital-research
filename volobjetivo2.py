"""
volobjetivo2.py — Lote V2: explotar el único método superviviente.
==================================================================
V1 (volatilidad objetivo) fue lo único que sobrevivió a ~70 pruebas. Este lote
NO busca edges nuevos: estira V1 para ver si puede dar MÁS RENTABILIDAD, no solo
menos riesgo. El benchmark honesto ya no es el mercado: es V1.

HALLAZGO PREVIO QUE OBLIGA A CORREGIR: el V1 publicado permite exposición hasta
2,0× SIN coste de financiación. Nadie presta gratis. La primera celda de este
lote es V1 con financiación real (tipo corto + 50 pb); todo lo demás se compara
contra ESA versión corregida. Si V1 no sobrevive a la financiación, eso también
se publica.

CONVENCIÓN UNIFORME DEL LOTE (aplicada a todas las variantes por igual):
  - El efectivo no invertido rinde el tipo corto (T-bill 13 semanas, ^IRX).
  - El apalancamiento paga tipo corto + 50 pb.
  - Coste de rebalanceo 0,05% sobre el cambio de exposición, como en el resto
    del laboratorio.

CATORCE CELDAS, RAZÓN ECONÓMICA ESCRITA ANTES DE MIRAR DATOS:

  1.  V1f — V1 corregido con financiación, vs S&P. ¿Sobrevive el resultado
      publicado cuando el apalancamiento cuesta dinero?
  2.  L100 — tope 1,0× (sin apalancar), vs S&P. Diagnóstico: ¿el mérito de V1
      está en apalancarse en calma o en reducirse en tormenta? Predicción: la
      caída máxima mejora igual; el retorno baja.
  3.  L150 — tope 1,5×, vs V1f. Moreira & Muir (2017): la gestión de volatilidad
      mejora Sharpe; el apalancamiento moderado es la única vía para convertir
      ese Sharpe en retorno. La celda con más probabilidad a priori del lote.
  4.  L200 — tope 2,0×, vs V1f. Sensibilidad al tope, no elección a posteriori.
  5.  TF — volatilidad objetivo + media de 10 meses (bajo la media: fuera, en
      efectivo al tipo corto), vs V1f. La volatilidad predice la MAGNITUD del
      riesgo; la tendencia, el RÉGIMEN. Señales parcialmente independientes.
  6.  SEMI — escalar por semivolatilidad (solo días negativos, 63 sesiones),
      vs V1f. El inversor teme caídas; la volatilidad de un rally no es riesgo
      y la total penaliza subidas violentas.
  7.  ASIM — reducir exposición de inmediato cuando la volatilidad sube,
      reconstruirla despacio (máx. +0,10×/mes), vs V1f. La volatilidad sube en
      saltos y decae despacio (asimetría documentada del GARCH): tras el pico
      conviene desconfiar un tiempo.
  8.  VOV — recorte adicional cuando la volatilidad-de-la-volatilidad es alta
      (solo recorta, nunca añade), vs V1f. Si el pronóstico de volatilidad es
      incierto, el escalado tiene error: menos confianza, menos exposición.
  9.  VDEF — volatilidad objetivo sobre mezcla defensiva 50/25/25
      (bolsa/oro/bonos), vs la mezcla sin gestionar. Base más estable → mejor
      estimación de volatilidad → menos error de escalado.
  10. V64 — volatilidad objetivo sobre 60/40, vs 60/40 sin gestionar. Ídem.
  11. EWMA — estimador EWMA (λ=0,94, RiskMetrics) en vez de ventana 21d,
      vs V1f. Robustez del núcleo: ¿depende el resultado del estimador?
  12. RV63 — ventana de 63 días en vez de 21, vs V1f. Ídem: más estable,
      más lenta.
  13. T10 — objetivo 10% anual, vs V1f. }  Sensibilidad al parámetro, declarada
  14. T15 — objetivo 15% anual, vs V1f. }  como tal: NO se elige el mejor luego.

PREDICCIONES ESCRITAS ANTES DE CORRER (se juzgan al final):
  - Solo L150/L200 pueden dar más retorno que V1f, y aun así con más caída.
  - L100 conservará casi toda la mejora de caída máxima: el mérito está en
    reducirse, no en apalancarse.
  - ASIM y VOV mejorarán caída como mucho; SEMI, empate; estimadores y
    objetivos, robustez sin ganador claro.

CONTROL NEGATIVO: la misma tubería de L150 con la señal de volatilidad
permutada al azar. Si "funciona" con ruido, la tubería está rota.

FDR conjunto de Benjamini-Hochberg sobre las 14 celdas. Estas celdas se suman
al contador acumulado del laboratorio (~70 → ~85): cualquier brillo aislado
merece desconfianza por multiple-testing acumulado. NO es asesoramiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
COSTE = 0.05                  # % por cambio de exposición
SPREAD_FIN = 0.005            # 50 pb sobre el tipo corto para apalancarse
VOL_OBJETIVO = 0.12
RF_FALLBACK = 0.03            # si no hay ^IRX, tipo corto constante (se avisa)
ACTIVOS = ["SPY", "GLD", "TLT", "IEF"]
N_CELDAS_DECLARADAS = 14


# ---------------------------------------------------------------- datos ----
def _cargar(sintetico=False):
    """Devuelve (precios diarios, tipo corto anualizado mensual) o (None, None)."""
    if sintetico:
        rng = np.random.default_rng(91)
        n = 3200
        idx = pd.bdate_range("2013-01-01", periods=n)
        out = {}
        for i, tk in enumerate(ACTIVOS):
            vol = 0.008 + 0.004 * (i % 3)
            out[tk] = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
        px = pd.DataFrame(out, index=idx)
        rf = pd.Series(RF_FALLBACK, index=px.resample("ME").last().index)
        return px, rf
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    try:
        df = yf.download(ACTIVOS, start=inicio, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
    except Exception:
        return None, None
    out = {}
    for tk in ACTIVOS:
        try:
            s = df[tk]["Close"].dropna()
            if len(s) > 1000:
                out[tk] = s
        except Exception:
            continue
    if "SPY" not in out:
        return None, None
    px = pd.DataFrame(out).dropna()
    # Tipo corto: T-bill 13 semanas. Si falla, constante con aviso.
    rf = None
    try:
        irx = yf.download("^IRX", start=inicio, auto_adjust=True, progress=False)["Close"].dropna()
        if len(irx) > 500:
            rf = (irx / 100.0).resample("ME").last().reindex(
                px.resample("ME").last().index).ffill().squeeze()
    except Exception:
        rf = None
    if rf is None:
        rf = pd.Series(RF_FALLBACK, index=px.resample("ME").last().index)
        rf.attrs["fallback"] = True
    return px, rf


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
                      f"<b>Sharpe {me['sharpe']} vs {mb['sharpe']}</b> (benchmark: {bench_txt})."),
            "_sh": me["sharpe"], "_shb": mb["sharpe"], "_cagr": me["cagr"], "_dd": me["dd"]}


# ---------------------------------------------------------------- motor ----
def _estimador_vol(dia, tipo):
    """Serie mensual de volatilidad anualizada prevista, SOLO con información
    pasada (shift(1) tras el remuestreo)."""
    if tipo == "rv21":
        v = dia.rolling(21).std() * np.sqrt(252)
    elif tipo == "rv63":
        v = dia.rolling(63).std() * np.sqrt(252)
    elif tipo == "ewma":
        lam = 0.94
        v = np.sqrt(dia.pow(2).ewm(alpha=1 - lam, adjust=False).mean() * 252)
    elif tipo == "semi":
        # Normalizada por √2: bajo simetría, semivol ≈ vol/√2. Sin esta corrección
        # la celda mediría "más apalancamiento medio", no el timing de la señal
        # (lo delató el control sintético: +0,39% "significativo" en ruido puro).
        neg = dia.clip(upper=0.0)
        v = np.sqrt(neg.pow(2).rolling(63).mean() * 252 * 2.0)
    else:
        raise ValueError(tipo)
    return v.resample("ME").last().shift(1)


def _estrategia(ret_men_base, expo, rf_men, coste=COSTE, spread=SPREAD_FIN):
    """Retorno mensual: expo × base + (1−expo) × efectivo − financiación − rotación.
    Con expo>1 la parte (1−expo) es negativa: se paga el tipo corto, y el spread
    se añade aparte sobre el exceso apalancado."""
    expo = expo.reindex(ret_men_base.index)
    rf_m = rf_men.reindex(ret_men_base.index).ffill() / 12.0
    apal = (expo - 1.0).clip(lower=0.0)
    camb = expo.diff().abs().fillna(0)
    r = (ret_men_base * expo + (1.0 - expo) * rf_m
         - apal * (spread / 12.0) - camb * (coste / 100.0))
    return r.dropna()


def _expo_asimetrica(expo, paso=0.10):
    """Baja inmediata, subida limitada a `paso` por mes."""
    vals = expo.values.copy()
    out = np.full_like(vals, np.nan)
    prev = np.nan
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            prev = np.nan
            continue
        if not np.isfinite(prev):
            out[i] = v
        elif v < prev:
            out[i] = v
        else:
            out[i] = min(v, prev + paso)
        prev = out[i]
    return pd.Series(out, index=expo.index)


# --------------------------------------------------------------- lote ----
def evaluar(sintetico=False):
    px, rf = _cargar(sintetico)
    if px is None:
        return None
    dia = px.pct_change()
    men = px.resample("ME").last()
    ret = men.pct_change().dropna(how="all")
    if "SPY" not in ret.columns or len(ret) < 80:
        return None
    spy = ret["SPY"]
    rf_fb = bool(getattr(rf, "attrs", {}).get("fallback", False))

    vol_spy = {t: _estimador_vol(dia["SPY"], t) for t in ("rv21", "rv63", "ewma", "semi")}

    def expo_de(vol, target=VOL_OBJETIVO, lo=0.3, hi=2.0):
        return (target / vol).clip(lo, hi)

    bloques = []
    seed = 200

    # 1 · V1f: V1 con financiación real, vs S&P
    e_v1f = expo_de(vol_spy["rv21"])
    v1f = _estrategia(spy, e_v1f, rf)
    b = _fila("1 · V1f — V1 con coste de financiación (tope 2,0×)",
              ("El V1 publicado se apalancaba hasta 2,0× gratis. Nadie presta gratis: aquí el "
               "apalancamiento paga tipo corto + 50 pb y el efectivo no invertido rinde el tipo "
               "corto. Si el resultado de V1 no sobrevive a esto, se publica igual. Todas las "
               "variantes del lote se comparan contra ESTA versión corregida."),
              "#5fb7c4", v1f, spy.reindex(v1f.index), "S&P 500", seed)
    if b is None:
        return None
    bloques.append(b); seed += 1
    ref = v1f  # benchmark del resto del lote

    # 2 · L100: sin apalancar, vs S&P
    l100 = _estrategia(spy, expo_de(vol_spy["rv21"], hi=1.0), rf)
    b = _fila("2 · L100 — sin apalancamiento (tope 1,0×)",
              ("Diagnóstico: ¿el mérito de V1 está en apalancarse en calma o en reducirse en "
               "tormenta? Predicción previa: la caída máxima mejora igual y el retorno baja. "
               "Si se cumple, el valor del método es defensivo, no ofensivo."),
              "#6ec08a", l100, spy.reindex(l100.index), "S&P 500", seed)
    if b: bloques.append(b)
    seed += 1

    # 3-4 · L150 / L200 vs V1f
    for tope, nombre, razon in (
        (1.5, "3 · L150 — tope 1,5×",
         "Moreira & Muir (2017): la gestión de volatilidad mejora el Sharpe, y el apalancamiento "
         "moderado es la única vía para convertirlo en retorno. La celda con más probabilidad "
         "a priori del lote."),
        (2.0, "4 · L200 — tope 2,0× (= V1f; mide el efecto del tope frente a 1,5×)",
         "Sensibilidad al tope de apalancamiento, declarada de antemano: no se elige el mejor "
         "tope a posteriori."),
    ):
        est = _estrategia(spy, expo_de(vol_spy["rv21"], hi=tope), rf)
        b = _fila(nombre, razon, "#e8b23a", est, ref.reindex(est.index).dropna(), "V1f", seed)
        if b: bloques.append(b)
        seed += 1

    # 5 · TF: filtro de tendencia de 10 meses
    ma10 = men["SPY"].rolling(10).mean().shift(1)
    dentro = (men["SPY"].shift(1) > ma10).astype(float)
    e_tf = (expo_de(vol_spy["rv21"]) * dentro.reindex(expo_de(vol_spy["rv21"]).index)).fillna(0)
    tf = _estrategia(spy, e_tf, rf)
    b = _fila("5 · TF — volatilidad objetivo + media de 10 meses",
              ("La volatilidad predice la MAGNITUD del riesgo; la media de 10 meses, el RÉGIMEN. "
               "Bajo la media: fuera del mercado, en efectivo al tipo corto. Señales parcialmente "
               "independientes; la combinación debería recortar las caídas largas."),
              "#b48ad6", tf, ref.reindex(tf.index).dropna(), "V1f", seed)
    if b: bloques.append(b)
    seed += 1

    # 6 · SEMI
    semi = _estrategia(spy, expo_de(vol_spy["semi"]), rf)
    b = _fila("6 · SEMI — escalar por semivolatilidad (solo días negativos)",
              ("El inversor teme caídas: la volatilidad de un rally no es riesgo, y la volatilidad "
               "total penaliza subidas violentas reduciendo exposición justo cuando el mercado "
               "corre. La semivolatilidad (63 sesiones) solo mira los días rojos. Normalizada por "
               "√2 para que quede en la escala de la vol total: así la celda mide el <i>timing</i> "
               "de la señal, no un apalancamiento medio mayor (fallo detectado y corregido con el "
               "control sintético antes de tocar datos reales)."),
              "#d4736a", semi, ref.reindex(semi.index).dropna(), "V1f", seed)
    if b: bloques.append(b)
    seed += 1

    # 7 · ASIM
    asim = _estrategia(spy, _expo_asimetrica(expo_de(vol_spy["rv21"])), rf)
    b = _fila("7 · ASIM — bajar rápido, subir despacio (máx. +0,10×/mes)",
              ("La volatilidad sube en saltos y decae despacio (asimetría documentada del GARCH): "
               "tras un pico de pánico conviene desconfiar un tiempo aunque el estimador ya se "
               "haya calmado. La exposición se recorta de inmediato pero se reconstruye gradual."),
              "#7a9fd4", asim, ref.reindex(asim.index).dropna(), "V1f", seed)
    if b: bloques.append(b)
    seed += 1

    # 8 · VOV
    rv_dia = dia["SPY"].rolling(21).std() * np.sqrt(252)
    vov = rv_dia.rolling(63).std()
    med = vov.rolling(252).median()
    factor = (med / vov).clip(0.5, 1.0).resample("ME").last().shift(1)
    e_vov = expo_de(vol_spy["rv21"]) * factor
    vv = _estrategia(spy, e_vov.dropna(), rf)
    b = _fila("8 · VOV — recorte extra cuando la volatilidad-de-la-volatilidad es alta",
              ("Si el propio pronóstico de volatilidad es incierto, el escalado tiene error de "
               "estimación. Cuando la vol-de-la-vol supera su mediana móvil, se recorta la "
               "exposición en proporción (solo recorta, nunca añade)."),
              "#8fbf9f", vv, ref.reindex(vv.index).dropna(), "V1f", seed)
    if b: bloques.append(b)
    seed += 1

    # 9-10 · Targeting sobre carteras diversificadas
    for cols_pesos, nombre, bench_nombre, razon in (
        ({"SPY": 0.50, "GLD": 0.25, "TLT": 0.25},
         "9 · VDEF — volatilidad objetivo sobre mezcla defensiva 50/25/25",
         "mezcla 50/25/25 sin gestionar",
         "Una base diversificada tiene volatilidad más estable que el índice puro: el estimador "
         "acierta más y el escalado comete menos error. Se compara contra la MISMA mezcla sin "
         "gestionar: el efecto medido es solo el del targeting."),
        ({"SPY": 0.60, "TLT": 0.40},
         "10 · V64 — volatilidad objetivo sobre cartera 60/40",
         "60/40 sin gestionar",
         "Mismo razonamiento sobre la cartera clásica 60/40. Si el targeting añade valor sobre "
         "bases distintas, el mecanismo es general y no un accidente del S&P."),
    ):
        cols = [c for c in cols_pesos if c in ret.columns]
        if len(cols) < len(cols_pesos):
            continue
        w = pd.Series(cols_pesos)
        base_dia = (dia[cols] * w).sum(axis=1, min_count=len(cols))
        base_men = (ret[cols] * w).sum(axis=1, min_count=len(cols)).dropna()
        vol_b = _estimador_vol(base_dia, "rv21")
        est = _estrategia(base_men, expo_de(vol_b, target=0.10), rf)
        b = _fila(nombre, razon, "#c9a86a", est, base_men.reindex(est.index),
                  bench_nombre, seed)
        if b: bloques.append(b)
        seed += 1

    # 11-12 · Estimadores alternativos
    for tipo, nombre, razon in (
        ("ewma", "11 · EWMA — estimador RiskMetrics (λ=0,94)",
         "Robustez del núcleo: si el resultado depende del estimador de volatilidad concreto, "
         "es frágil. El EWMA pondera más lo reciente y reacciona antes que la ventana fija."),
        ("rv63", "12 · RV63 — ventana de 63 días",
         "El caso contrario: estimador más lento y estable. Entre EWMA, RV21 y RV63 no se "
         "elige ganador: se mide si los tres cuentan la misma historia."),
    ):
        est = _estrategia(spy, expo_de(vol_spy[tipo]), rf)
        b = _fila(nombre, razon, "#9a8fb8", est, ref.reindex(est.index).dropna(), "V1f", seed)
        if b: bloques.append(b)
        seed += 1

    # 13-14 · Sensibilidad del objetivo
    for tgt, nombre in ((0.10, "13 · T10 — objetivo 10% anual"),
                        (0.15, "14 · T15 — objetivo 15% anual")):
        est = _estrategia(spy, expo_de(vol_spy["rv21"], target=tgt), rf)
        b = _fila(nombre,
                  "Sensibilidad al parámetro del objetivo, declarada como tal ANTES de correr: "
                  "no se elige el mejor objetivo a posteriori. Solo se mide cuánto cambia el "
                  "resultado al moverlo.",
                  "#7d8a99", est, ref.reindex(est.index).dropna(), "V1f", seed)
        if b: bloques.append(b)
        seed += 1

    if not bloques:
        return None

    # Control negativo: L150 con la señal de volatilidad permutada.
    ctrl_txt = ""
    try:
        rng = np.random.default_rng(314)
        v = vol_spy["rv21"].dropna()
        v_perm = pd.Series(rng.permutation(v.values), index=v.index)
        est_c = _estrategia(spy, expo_de(v_perm, hi=1.5), rf)
        dfc = pd.DataFrame({"e": est_c, "b": ref}).dropna()
        rc = _boot((dfc["e"].values - dfc["b"].values) * 100, seed=999)
        if rc:
            ctrl_txt = (f"<br><b>Control negativo:</b> la misma tubería de L150 con la señal de "
                        f"volatilidad permutada al azar da un exceso de {rc['m']:+.3f}% mensual "
                        f"frente a V1f (p={rc['p']}). Si esto saliera significativo y positivo, la "
                        f"tubería estaría rota; se espera ruido puro.")
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

    aviso_rf = ("" if not rf_fb else
                "<br><b>Aviso:</b> no se pudo descargar el T-bill (^IRX); se usó un tipo corto "
                "constante del 3% anual. Los niveles absolutos cambian algo; las comparaciones "
                "relativas, poco.")

    return {
        "id": "vol_lote2",
        "etiqueta": "Lote V2 · Explotar la volatilidad objetivo",
        "tipo": f"{len(bloques)} de {N_CELDAS_DECLARADAS} celdas declaradas · FDR conjunto · neto de costes y financiación",
        "modelo": "volobj",
        "figuras_panel": True,
        "intro": (
            "V1 fue lo único que sobrevivió a ~70 pruebas. Este lote no busca edges nuevos: "
            "estira V1 para ver si puede dar más <b>rentabilidad</b>, no solo menos riesgo. El "
            "benchmark ya no es el mercado: es V1 corregido con coste de financiación real "
            "(el V1 original se apalancaba gratis hasta 2,0× — ese fallo se corrige aquí y se "
            "publica). Convención uniforme: el efectivo rinde el tipo corto, apalancarse paga "
            "tipo corto + 50 pb, rebalancear cuesta 0,05%. Predicciones escritas antes de "
            "correr: solo L150/L200 pueden dar más retorno; L100 conservará la mejora de caída "
            "(el mérito es defensivo); ASIM y VOV mejorarán caída como mucho; el resto, "
            "robustez sin ganador." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Corrección conjunta sobre {len(pvals)} celdas (Benjamini-Hochberg, FDR 10%): "
                     f"{n_fdr} sobreviven. <b>Contexto:</b> con este lote el laboratorio supera las "
                     f"~85 pruebas acumuladas. A ese volumen, un brillo aislado es sospechoso por "
                     f"defecto: la vara para creerse algo nuevo sube con cada prueba."),
        "nota": (ctrl_txt + aviso_rf +
                 "<br>Ninguna celda predice dirección. Todo el lote explota un solo hecho ya "
                 "demostrado: la volatilidad es predecible. La pregunta es si eso puede pagar "
                 "más que el mercado, o solo doler menos. No es recomendación de inversión."),
    }
