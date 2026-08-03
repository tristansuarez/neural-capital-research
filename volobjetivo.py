"""
volobjetivo.py — Tres hipótesis nuevas, cada una con su razón previa.
=====================================================================

  V1. VOLATILIDAD OBJETIVO. Escalar la exposición para mantener la volatilidad de
      la cartera constante: cuando el mercado se agita, reducir; cuando se calma,
      aumentar. RAZÓN: no requiere predecir dirección, solo volatilidad — y este
      laboratorio ya demostró que la volatilidad SÍ es predecible (GARCH con
      correlación 0,93 frente a la realizada). Es la aplicación directa del único
      edge sólido encontrado. Además la volatilidad agrupa (los días agitados van
      juntos), así que la señal es persistente.

  V2. MOMENTUM ENTRE CLASES DE ACTIVO. Cada mes, mantener las 3 clases con mejor
      momentum a 12 meses de entre 9 disponibles. RAZÓN: mecanismo distinto al
      momentum de acciones individuales (que ya falló aquí). Entre clases, el
      momentum refleja flujos macro lentos —rotación de capital entre bolsa, bonos
      y materias primas— que tardan meses en completarse.

  V3. DEGRADACIÓN DE FACTORES. ¿Rinden menos los factores clásicos DESPUÉS de
      publicarse? RAZÓN: si un método se arbitra al conocerse, su retorno debería
      caer tras la fecha de publicación. Es la tesis de McLean & Pontiff (2016).
      No es una estrategia: es medir por qué las estrategias dejan de funcionar.

Todas con FDR conjunto y juzgadas por Sharpe además de retorno. NO es asesoramiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
COSTE = 0.05
ACTIVOS = ["SPY", "GLD", "TLT", "IEF", "VGK", "EEM", "EWJ", "VNQ", "DBC"]
VOL_OBJETIVO = 0.12      # 12% anual, típico de una cartera equilibrada


def _cargar(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(91)
        n = 3200
        idx = pd.bdate_range("2013-01-01", periods=n)
        out = {}
        for i, tk in enumerate(ACTIVOS):
            vol = 0.008 + 0.004 * (i % 3)
            out[tk] = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
        return pd.DataFrame(out, index=idx)
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    try:
        df = yf.download(ACTIVOS, start=inicio, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
    except Exception:
        return None
    out = {}
    for tk in ACTIVOS:
        try:
            s = df[tk]["Close"].dropna()
            if len(s) > 1000:
                out[tk] = s
        except Exception:
            continue
    return pd.DataFrame(out).dropna() if "SPY" in out and len(out) >= 4 else None


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


def _fila(nombre, razon, color, est, bh, extra_txt=""):
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
                      f"<b>Sharpe {me['sharpe']} vs {mb['sharpe']}</b>. {extra_txt}"),
            "_sh": me["sharpe"], "_shb": mb["sharpe"], "_cagr": me["cagr"], "_dd": me["dd"]}


def evaluar(sintetico=False):
    px = _cargar(sintetico)
    if px is None:
        return None
    dia = px.pct_change()
    men = px.resample("ME").last()
    ret = men.pct_change().dropna(how="all")
    if "SPY" not in ret.columns or len(ret) < 80:
        return None
    spy = ret["SPY"]
    bloques = []

    # --- V1: volatilidad objetivo sobre el S&P ---
    vol_real = dia["SPY"].rolling(21).std() * np.sqrt(252)
    vol_m = vol_real.resample("ME").last().shift(1)          # solo información pasada
    exposicion = (VOL_OBJETIVO / vol_m).clip(0.3, 2.0)        # límites realistas
    camb = exposicion.diff().abs().fillna(0)
    est_v1 = (spy * exposicion - camb * (COSTE / 100.0)).dropna()
    expo_media = float(exposicion.reindex(est_v1.index).mean())
    b = _fila("V1 · Volatilidad objetivo (12% anual)",
              ("Escalar la exposición para mantener la volatilidad constante: reducir cuando el "
               "mercado se agita, aumentar cuando se calma. No predice dirección, solo volatilidad "
               "—y este laboratorio ya demostró que la volatilidad SÍ es predecible (GARCH, "
               "correlación 0,93)—. Es la aplicación directa del único edge sólido encontrado."),
              "#5fb7c4", est_v1, spy.reindex(est_v1.index),
              f"Exposición media {expo_media:.2f}× (1,0 = totalmente invertido).")
    if b:
        bloques.append(b)

    # --- V2: momentum entre clases de activo ---
    disp = [c for c in ACTIVOS if c in ret.columns]
    if len(disp) >= 5:
        mom = men[disp] / men[disp].shift(12) - 1.0
        filas = []
        for i in range(13, len(ret)):
            f = ret.index[i]
            señal = mom.reindex([ret.index[i - 1]])[disp]
            if señal.isna().all(axis=None):
                continue
            s = señal.iloc[0].dropna()
            if len(s) < 4:
                continue
            top = s.sort_values(ascending=False).head(3).index
            r = ret.loc[f, top].mean()
            if np.isfinite(r):
                filas.append((f, float(r) - COSTE / 100.0))
        if len(filas) > 60:
            est_v2 = pd.Series([r for _f, r in filas], index=[f for f, _r in filas])
            eq_pond = ret[disp].mean(axis=1).reindex(est_v2.index)
            b = _fila("V2 · Momentum entre clases de activo (top 3 de 9)",
                      ("Cada mes, mantener las 3 clases con mejor momentum a 12 meses. Mecanismo "
                       "distinto al momentum de acciones individuales (que ya falló aquí): entre "
                       "clases refleja flujos macro lentos —rotación de capital entre bolsa, bonos y "
                       "materias primas— que tardan meses en completarse."),
                      "#6ec08a", est_v2, eq_pond,
                      "Se compara contra repartir por igual entre las 9 clases.")
            if b:
                bloques.append(b)

    if not bloques:
        return None

    # --- V3: degradación de factores (diagnóstico, no estrategia) ---
    # Momentum de acciones publicado en 1993 (Jegadeesh-Titman); valor en 1992
    # (Fama-French). Comparamos el momentum entre clases antes/después de 2010,
    # cuando los ETFs de factores se popularizaron y el arbitraje se hizo masivo.
    deg = ""
    if len(bloques) > 1 and "_sh" in bloques[-1]:
        est_v2 = pd.Series(dtype=float)
    try:
        if len(disp) >= 5 and len(filas) > 60:
            serie = pd.Series([r for _f, r in filas], index=[f for f, _r in filas])
            base = ret[disp].mean(axis=1).reindex(serie.index)
            ex = (serie - base) * 100
            antes = ex[ex.index < "2013-01-01"]
            despues = ex[ex.index >= "2013-01-01"]
            if len(antes) >= 24 and len(despues) >= 24:
                ra, rd = _boot(antes.values), _boot(despues.values, seed=101)
                if ra and rd:
                    deg = (
                        "<br><br><b>V3 · ¿Se degrada el factor al popularizarse?</b> Los ETFs de "
                        "factores se masificaron hacia 2013. Si un método se arbitra al conocerse, "
                        "su ventaja debería caer después. Es la tesis de McLean &amp; Pontiff (2016), "
                        "medida aquí con datos propios."
                        "<div class='ops-scroll'><table class='ops'><thead><tr><th>Periodo</th>"
                        "<th>Ventaja mensual</th><th>IC 90%</th><th>p</th><th>meses</th></tr></thead>"
                        f"<tbody><tr><td>Antes de 2013</td>"
                        f"<td class='{'pos' if ra['m'] > 0 else 'neg'}'>{ra['m']:+.3f}%</td>"
                        f"<td class='est-obs'>{ra['ic']}</td><td>{ra['p']}</td>"
                        f"<td class='est-obs'>{ra['n']}</td></tr>"
                        f"<tr><td>Desde 2013</td>"
                        f"<td class='{'pos' if rd['m'] > 0 else 'neg'}'>{rd['m']:+.3f}%</td>"
                        f"<td class='est-obs'>{rd['ic']}</td><td>{rd['p']}</td>"
                        f"<td class='est-obs'>{rd['n']}</td></tr></tbody></table></div>"
                        f"<div class='ch-sub' style='margin-top:8px'>Diferencia: "
                        f"<b>{rd['m'] - ra['m']:+.3f}% mensual</b>. Si es claramente negativa, apoya "
                        f"la hipótesis de que el factor se arbitró al popularizarse. Ojo: con dos "
                        f"submuestras cortas, esto es indicativo, no concluyente.</div>")
    except Exception:
        deg = ""

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
        "id": "vol_objetivo",
        "etiqueta": "Volatilidad objetivo y momentum entre clases",
        "tipo": f"{len(bloques)} hipótesis nuevas · FDR conjunto · neto de costes",
        "modelo": "volobj",
        "figuras_panel": True,
        "intro": ("Tres hipótesis que no se habían probado en este laboratorio, cada una con su razón "
                  "económica escrita antes de medir. La primera aplica el único edge demostrado aquí "
                  "(el GARCH predice volatilidad); la segunda usa un mecanismo distinto al que ya "
                  "falló; la tercera mide por qué los métodos dejan de funcionar." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Corrección conjunta sobre {len(pvals)} hipótesis (Benjamini-Hochberg, FDR 10%): "
                     f"{n_fdr} sobreviven. <b>Contexto importante:</b> este laboratorio lleva del orden "
                     f"de 70 pruebas acumuladas. A ese volumen, incluso corrigiendo dentro de cada "
                     f"módulo, un resultado aislado que destaque debe mirarse con desconfianza: la "
                     f"probabilidad de falsos positivos acumulados es alta."),
        "nota": (deg + "<br>Ninguna de estas hipótesis predice la dirección del mercado. V1 escala "
                 "exposición según volatilidad (predecible), V2 sigue flujos macro lentos y V3 es un "
                 "diagnóstico. No es recomendación de inversión."),
    }
