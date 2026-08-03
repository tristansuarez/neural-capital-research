"""
primas.py — Lote V3: primas de riesgo. La vía honesta hacia más rentabilidad.
=============================================================================
Tras ~85 pruebas, este laboratorio tiene un solo hecho útil (la volatilidad es
predecible) y una sola conclusión sobre rentabilidad: no se le puede sacar más
al S&P sin comprar más riesgo del S&P. Este lote cambia de pregunta: en vez de
predecir mejor, COBRAR MÁS PRIMAS. Una prima de riesgo no es una anomalía ni
una ineficiencia: es lo que el mercado paga de forma persistente a quien carga
con un riesgo que otros quieren quitarse de encima. No se arbitra del todo
porque cobrarla duele.

CONVENCIÓN UNIFORME (idéntica al lote V2): efectivo al tipo corto (^IRX),
apalancamiento a tipo corto + 50 pb, coste de rebalanceo 0,05%.

NUEVE CELDAS, RAZÓN ECONÓMICA ESCRITA ANTES DE MIRAR DATOS:

  1. BXM — venta de calls cubierta (índice CBOE BuyWrite) vs S&P. El seguro
     contra movimientos se paga con sobreprecio persistente porque el miedo es
     asimétrico: el vendedor de opciones cobra la prima y carga con el crash.
  2. PUT — venta de puts con colateral (índice CBOE PutWrite) vs S&P. La misma
     prima por la vía directa: vender el seguro de caída, que es el más caro.
  3. HY — bonos de alto rendimiento (HYG) vs Tesoro de duración comparable
     (mezcla 50/50 de 1-3 y 7-10 años). Cobrar por asumir riesgo de impago,
     mecanismo distinto de la duración.
  4. IG — crédito de grado de inversión (LQD) vs Tesoro 7-10 (IEF). La misma
     prima en su versión suave: menos impago, menos prima, menos correlación
     con la bolsa.
  5. DEF — sectores defensivos 50/50 consumo básico + utilities (XLP/XLU) vs
     S&P, tal cual. Base descriptiva de la celda 6.
  6. DEFβ — defensivos apalancados a beta 1 (exposición = 1/beta móvil, tope
     1,5×, financiado a tipo + 50 pb) vs S&P. Frazzini-Pedersen: las
     restricciones de apalancamiento inflan la beta alta; la beta baja queda
     estructuralmente barata, y solo se cobra apalancándola.
  7. COMBO — cartera equiponderada de primas disponibles (bolsa + venta de
     puts + alto rendimiento + defensivos), rebalanceo mensual, vs S&P. Si las
     primas cobran por riesgos DISTINTOS, sus crashes no coinciden del todo.
  8. COMBO-V — la cartera de primas con volatilidad objetivo 12% encima
     (tope 1,5×, financiado), vs V1f. La celda central del lote: cobrar más
     primas Y gestionar el único riesgo predecible. Es la única construcción
     con derecho a priori a más rentabilidad con Sharpe igual o mejor.
  9. COMBO-V vs 60/40 estático. El competidor sin fricción: si una cartera
     fija iguala esto sin rotación mensual ni complejidad, la construcción
     entera sobra. Benchmark deliberadamente incómodo.

PREDICCIONES ESCRITAS ANTES DE CORRER:
  - BXM/PUT: prima real pero con caídas brutales concentradas (feb-2018,
    mar-2020); la duda no es si paga, es si sobrevive a sus colas.
  - HY: prima positiva pero correlacionada con la bolsa justo cuando duele;
    puede no añadir nada en cartera.
  - IG: prima pequeña, aporte marginal.
  - DEF crudo: menos retorno y menos riesgo (es beta baja sin apalancar).
    DEFβ es el test real; si tras financiación no queda nada, se publica.
  - COMBO/COMBO-V: la única vía plausible a más CAGR que el mercado con
    caída menor. Si ni esto lo logra, la respuesta del laboratorio a "más
    rentabilidad" queda cerrada en negativo con todas las letras.

CONTROL NEGATIVO: la tubería de COMBO-V con la señal de volatilidad permutada.

FDR conjunto de Benjamini-Hochberg sobre las 9 celdas. El contador acumulado
del laboratorio pasa de ~85 a ~95: todo brillo aislado, bajo sospecha. Lo que
sobreviva aquí NO se cree hasta pasar por el forward-test. NO es asesoramiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
COSTE = 0.05
SPREAD_FIN = 0.005
RF_FALLBACK = 0.03
VOL_OBJETIVO = 0.12
N_CELDAS_DECLARADAS = 9

TICKERS = ["SPY", "HYG", "LQD", "IEF", "SHY", "XLP", "XLU"]
INDICES = ["^BXM", "^PUT"]


# ---------------------------------------------------------------- datos ----
def _cargar(sintetico=False):
    """(precios diarios de tickers+índices, tipo corto mensual) o (None, None)."""
    if sintetico:
        rng = np.random.default_rng(47)
        n = 3200
        idx = pd.bdate_range("2013-01-01", periods=n)
        base = rng.normal(0.0003, 0.010, n)                      # factor bolsa
        out = {}
        for i, tk in enumerate(TICKERS + INDICES):
            beta = [1.0, 0.4, 0.2, -0.1, 0.0, 0.6, 0.5, 0.7, 0.7][i]
            eps = rng.normal(0.0001, 0.004, n)
            out[tk] = 100 * np.exp(np.cumsum(beta * base + eps))
        px = pd.DataFrame(out, index=idx)
        rf = pd.Series(RF_FALLBACK, index=px.resample("ME").last().index)
        return px, rf
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
        return None, None
    for tk in INDICES:   # los índices CBOE van uno a uno: fallan con más frecuencia
        try:
            s = yf.download(tk, start=inicio, auto_adjust=True,
                            progress=False)["Close"].dropna().squeeze()
            if len(s) > 1000:
                out[tk] = s
        except Exception:
            continue
    if "SPY" not in out:
        return None, None
    px = pd.DataFrame(out).sort_index()
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
            "_sh": me["sharpe"], "_dd": me["dd"]}


# ---------------------------------------------------------------- motor ----
def _vol_estimada(dia):
    return (dia.rolling(21).std() * np.sqrt(252)).resample("ME").last().shift(1)


def _con_gestion(ret_men, expo, rf_men, coste=COSTE, spread=SPREAD_FIN):
    expo = expo.reindex(ret_men.index)
    rf_m = rf_men.reindex(ret_men.index).ffill() / 12.0
    apal = (expo - 1.0).clip(lower=0.0)
    camb = expo.diff().abs().fillna(0)
    return (ret_men * expo + (1.0 - expo) * rf_m
            - apal * (spread / 12.0) - camb * (coste / 100.0)).dropna()


def _mezcla(ret, pesos, coste=COSTE):
    cols = [c for c in pesos if c in ret.columns]
    if len(cols) < len(pesos):
        return None
    w = pd.Series({c: pesos[c] for c in cols})
    w = w / w.sum()
    r = (ret[cols] * w).sum(axis=1, min_count=len(cols))
    return (r - coste / 100.0 * 0.5).dropna()   # rotación media de un rebalanceo fijo


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
    tiene = lambda *c: all(x in ret.columns for x in c)

    bloques, faltan = [], []
    seed = 300

    # 1-2 · Prima de varianza
    for tk, nombre, razon in (
        ("^BXM", "1 · BXM — venta de calls cubierta (índice CBOE) ",
         "El seguro contra movimientos se paga con sobreprecio persistente porque el miedo es "
         "asimétrico: quien vende la opción cobra la prima de varianza y carga con el crash. "
         "BXM la cobra vendiendo calls sobre cartera comprada."),
        ("^PUT", "2 · PUT — venta de puts con colateral (índice CBOE)",
         "La misma prima por la vía directa: vender el seguro de caída, que es el que más "
         "sobreprecio lleva. Predicción previa: prima real, pero con caídas brutales "
         "concentradas (feb-2018, mar-2020) — la duda no es si paga, es si se sobrevive."),
    ):
        if tk in ret.columns:
            b = _fila(nombre, razon, "#d4736a", ret[tk].dropna(),
                      spy.reindex(ret[tk].dropna().index), "S&P 500", seed)
            if b: bloques.append(b)
        else:
            faltan.append(tk)
        seed += 1

    # 3 · Alto rendimiento vs Tesoro de duración comparable
    if tiene("HYG", "SHY", "IEF"):
        teso = (ret["SHY"] * 0.5 + ret["IEF"] * 0.5).dropna()
        b = _fila("3 · HY — alto rendimiento (HYG) vs Tesoro de duración comparable",
                  ("Cobrar por asumir riesgo de impago, mecanismo distinto de la duración. Se "
                   "compara contra una mezcla 50/50 de Tesoro 1-3 y 7-10 años para aislar la "
                   "prima de crédito del efecto tipos. Predicción: prima positiva pero "
                   "correlacionada con la bolsa justo cuando duele."),
                  "#e8b23a", ret["HYG"].dropna(), teso, "Tesoro duración similar", seed)
        if b: bloques.append(b)
    else:
        faltan.append("HYG/SHY/IEF")
    seed += 1

    # 4 · Grado de inversión
    if tiene("LQD", "IEF"):
        b = _fila("4 · IG — crédito grado de inversión (LQD) vs Tesoro 7-10 (IEF)",
                  ("La prima de crédito en versión suave: menos impago, menos prima, menos "
                   "correlación con la bolsa. Predicción: aporte marginal."),
                  "#c9a86a", ret["LQD"].dropna(),
                  ret["IEF"].reindex(ret["LQD"].dropna().index), "IEF", seed)
        if b: bloques.append(b)
    else:
        faltan.append("LQD/IEF")
    seed += 1

    # 5-6 · Beta baja: defensivos crudos y apalancados a beta 1
    defensa = None
    if tiene("XLP", "XLU"):
        defensa = (ret["XLP"] * 0.5 + ret["XLU"] * 0.5).dropna()
        b = _fila("5 · DEF — defensivos 50/50 consumo básico + utilities, tal cual",
                  ("Base descriptiva de la celda 6: beta baja sin apalancar. Predicción: menos "
                   "retorno y menos riesgo que el S&P — eso NO es la prima todavía."),
                  "#8fbf9f", defensa, spy.reindex(defensa.index), "S&P 500", seed)
        if b: bloques.append(b)
        seed += 1

        defensa_dia = (dia["XLP"] * 0.5 + dia["XLU"] * 0.5)
        cov = defensa_dia.rolling(252).cov(dia["SPY"])
        var = dia["SPY"].rolling(252).var()
        beta = (cov / var).resample("ME").last().shift(1)
        expo_b = (1.0 / beta).clip(1.0, 1.5)
        defb = _con_gestion(defensa, expo_b, rf)
        b = _fila("6 · DEFβ — defensivos apalancados a beta 1 (tope 1,5×, financiado)",
                  ("Frazzini-Pedersen: las restricciones de apalancamiento inflan los activos de "
                   "beta alta y dejan la beta baja estructuralmente barata — pero solo se cobra "
                   "apalancándola. Exposición = 1/beta móvil de 252 sesiones, financiada a tipo "
                   "corto + 50 pb. Si tras la financiación no queda nada, se publica."),
                  "#7a9fd4", defb, spy.reindex(defb.index), "S&P 500", seed)
        if b: bloques.append(b)
    else:
        faltan.append("XLP/XLU")
        seed += 1
    seed += 1

    # 7 · Cartera de primas
    combo = None
    put_col = "^PUT" if "^PUT" in ret.columns else ("^BXM" if "^BXM" in ret.columns else None)
    patas = {"SPY": 1.0}
    if put_col: patas[put_col] = 1.0
    if "HYG" in ret.columns: patas["HYG"] = 1.0
    if defensa is not None: patas["_DEF"] = 1.0
    if len(patas) >= 3:
        ret_ext = ret.copy()
        if defensa is not None:
            ret_ext["_DEF"] = defensa
        combo = _mezcla(ret_ext, patas)
        etiq = " + ".join(k.replace("_DEF", "defensivos").replace("^", "") for k in patas)
        b = _fila(f"7 · COMBO — cartera equiponderada de primas ({etiq})",
                  ("Si las primas cobran por riesgos DISTINTOS (mercado, varianza, impago, "
                   "beta baja), sus crashes no coinciden del todo y la mezcla cobra más de lo "
                   "que suma en riesgo. Rebalanceo mensual, coste incluido."),
                  "#b48ad6", combo, spy.reindex(combo.index), "S&P 500", seed)
        if b: bloques.append(b)
    seed += 1

    # 8-9 · COMBO con volatilidad objetivo, contra V1f y contra 60/40
    if combo is not None and len(combo) > 80:
        combo_dia = None
        cols_dia = [c for c in patas if c != "_DEF" and c in dia.columns]
        partes = [dia[c] for c in cols_dia]
        if defensa is not None:
            partes.append((dia["XLP"] * 0.5 + dia["XLU"] * 0.5))
        combo_dia = pd.concat(partes, axis=1).mean(axis=1)
        vol_c = _vol_estimada(combo_dia)
        expo_c = (VOL_OBJETIVO / vol_c).clip(0.3, 1.5)
        combo_v = _con_gestion(combo, expo_c, rf)

        # V1f de referencia, reconstruido igual que en el lote V2
        vol_spy = _vol_estimada(dia["SPY"])
        v1f = _con_gestion(spy, (VOL_OBJETIVO / vol_spy).clip(0.3, 2.0), rf)

        b = _fila("8 · COMBO-V — cartera de primas con volatilidad objetivo (tope 1,5×)",
                  ("La celda central del lote: cobrar varias primas Y gestionar el único riesgo "
                   "predecible. Es la única construcción con derecho a priori a más rentabilidad "
                   "con Sharpe igual o mejor. Benchmark: V1f, el superviviente del lote V2."),
                  "#5fb7c4", combo_v, v1f.reindex(combo_v.index).dropna(), "V1f", seed)
        if b: bloques.append(b)
        seed += 1

        if tiene("IEF"):
            b6040 = _mezcla(ret, {"SPY": 0.6, "IEF": 0.4})
            b = _fila("9 · COMBO-V vs cartera 60/40 estática",
                      ("El competidor sin fricción: sin rotación mensual, sin complejidad, sin "
                       "disciplina que mantener. Si el 60/40 fijo iguala a la construcción "
                       "entera, la construcción sobra. Benchmark deliberadamente incómodo."),
                      "#7d8a99", combo_v, b6040.reindex(combo_v.index).dropna(), "60/40", seed)
            if b: bloques.append(b)
        seed += 1
    else:
        seed += 2

    if not bloques:
        return None

    # Control negativo: COMBO-V con la señal de volatilidad permutada.
    ctrl_txt = ""
    try:
        if combo is not None:
            rng = np.random.default_rng(314)
            v = _vol_estimada(combo_dia).dropna()
            v_perm = pd.Series(rng.permutation(v.values), index=v.index)
            est_c = _con_gestion(combo, (VOL_OBJETIVO / v_perm).clip(0.3, 1.5), rf)
            dfc = pd.DataFrame({"e": est_c, "b": combo}).dropna()
            rc = _boot((dfc["e"].values - dfc["b"].values) * 100, seed=999)
            if rc:
                ctrl_txt = (f"<br><b>Control negativo:</b> COMBO-V con la señal de volatilidad "
                            f"permutada al azar da {rc['m']:+.3f}% mensual frente a la cartera sin "
                            f"gestionar (p={rc['p']}). Se espera ruido puro; si saliera "
                            f"significativo y positivo, la tubería estaría rota.")
    except Exception:
        ctrl_txt = ""

    # Diagnóstico de correlaciones entre primas (no es celda: es contexto).
    diag = ""
    try:
        ex = {}
        if put_col: ex["varianza"] = (ret[put_col] - spy)
        if "HYG" in ret.columns and "IEF" in ret.columns:
            ex["crédito"] = (ret["HYG"] - ret["IEF"])
        if defensa is not None:
            ex["beta baja"] = (defensa - spy)
        if len(ex) >= 2:
            co = pd.DataFrame(ex).dropna().corr().round(2)
            filas = "".join(
                f"<tr><td>{a}</td>" + "".join(
                    f"<td class='est-obs'>{co.loc[a, b]:+.2f}</td>" for b in co.columns) + "</tr>"
                for a in co.index)
            cab = "".join(f"<th>{c}</th>" for c in co.columns)
            diag = ("<br><b>Correlación entre los excesos de las primas</b> (contexto, no celda): "
                    "cuanto más cerca de cero, más real es la diversificación entre primas."
                    f"<div class='ops-scroll'><table class='ops'><thead><tr><th></th>{cab}</tr>"
                    f"</thead><tbody>{filas}</tbody></table></div>")
    except Exception:
        diag = ""

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
                 ". Las celdas afectadas no se evalúan (y se dice, en vez de rellenarlas).")
    if rf_fb:
        aviso += ("<br><b>Aviso:</b> sin ^IRX; tipo corto constante del 3% anual.")

    return {
        "id": "primas_riesgo",
        "etiqueta": "Lote V3 · Primas de riesgo",
        "tipo": f"{len(bloques)} de {N_CELDAS_DECLARADAS} celdas declaradas · FDR conjunto · neto de costes y financiación",
        "modelo": "volobj",
        "figuras_panel": True,
        "intro": (
            "Cambio de pregunta: en vez de predecir mejor, <b>cobrar más primas</b>. Una prima "
            "de riesgo no es una anomalía: es lo que el mercado paga de forma persistente a "
            "quien carga con un riesgo que otros quieren quitarse de encima — y no se arbitra "
            "del todo porque cobrarla duele. Cuatro familias (varianza, crédito, beta baja y su "
            "combinación con la volatilidad objetivo), nueve celdas, predicciones escritas antes "
            "de correr: BXM/PUT pagan pero sus colas son la duda; HY se correlaciona con la "
            "bolsa cuando duele; DEF crudo no es la prima (DEFβ sí es el test); y COMBO-V es la "
            "única construcción con derecho a priori a más rentabilidad. Benchmarks exigentes: "
            "V1f y el 60/40 estático, no solo el S&P." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Corrección conjunta sobre {len(pvals)} celdas (Benjamini-Hochberg, FDR 10%): "
                     f"{n_fdr} sobreviven. <b>Contexto:</b> el laboratorio supera las ~95 pruebas "
                     f"acumuladas. Lo que brille aquí no se cree hasta pasar por el forward-test: "
                     f"un backtest solo, a estas alturas, ya no demuestra nada."),
        "nota": (ctrl_txt + diag + aviso +
                 "<br>Las primas pagan por cargar con riesgos reales: el crash del vendedor de "
                 "puts, el impago del bonista, el rezago del defensivo en los rallies. Nada aquí "
                 "es dinero gratis y nada es recomendación de inversión."),
    }
