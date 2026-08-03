"""
cartera.py — Construcción de cartera: lo único que ha aportado en el laboratorio.
================================================================================
Cuatro hipótesis, ninguna predice nada. Todas se comparan contra el S&P 500 y
contra la cartera equiponderada simple, con coste y FDR conjunto.

  C1. REBALANCEO COMO FUENTE DE RETORNO. Rebalancear vende lo que subió y compra
      lo que bajó. Con activos volátiles y poco correlacionados eso genera retorno
      («volatility harvesting»): la media geométrica de la cartera rebalanceada
      supera a la media de las medias geométricas. Se compara la MISMA cartera con
      rebalanceo mensual, anual y sin rebalancear: la diferencia es el efecto puro.

  C2. PARIDAD DE RIESGO. Repartir para que cada activo aporte el MISMO riesgo, no
      el mismo capital. Razón: con pesos iguales por capital, la bolsa domina el
      riesgo total; igualando riesgo, la diversificación es real. Es el principio
      de las carteras «all weather».

  C3. DIVERSIFICACIÓN GEOGRÁFICA. Añadir Europa, emergentes y Japón. Razón: los
      ciclos económicos no coinciden entre regiones; la correlación imperfecta
      reduce riesgo sin sacrificar retorno esperado.

  C4. BONOS COMO ACTIVO. No como señal (eso ya falló), sino en cartera: su
      correlación negativa con la bolsa en crisis es lo que amortigua las caídas.

Se juzga por SHARPE y CAÍDA MÁXIMA además de por retorno: en este laboratorio el
valor ha estado siempre en el riesgo, no en la rentabilidad. NO es asesoramiento.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
COSTE = 0.05          # % por rebalanceo
UNIVERSO = {
    "SPY": "Bolsa EE.UU.", "GLD": "Oro", "TLT": "Bonos largos", "IEF": "Bonos medios",
    "VGK": "Europa", "EEM": "Emergentes", "EWJ": "Japón", "VNQ": "Inmobiliario",
    "DBC": "Materias primas",
}


def _cargar(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(73)
        n = 3200
        idx = pd.bdate_range("2013-01-01", periods=n)
        return pd.DataFrame({tk: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
                             for tk in UNIVERSO}, index=idx)
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    tks = list(UNIVERSO)
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
    if "SPY" not in out or len(out) < 4:
        return None
    return pd.DataFrame(out).dropna()


def _boot(x, seed=79, n_boot=3000):
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


def _cartera(ret, pesos, cada=1, coste=COSTE):
    """Retorno mensual de una cartera con rebalanceo cada 'cada' meses.
    cada=1 mensual, cada=12 anual, cada=0 sin rebalancear (deriva libre)."""
    cols = list(pesos.keys())
    w0 = np.array([pesos[c] for c in cols], float)
    w0 = w0 / w0.sum()
    R = ret[cols].values
    n = len(R)
    out = np.full(n, np.nan)
    w = w0.copy()
    for t in range(n):
        r = R[t]
        if not np.isfinite(r).all():
            continue
        out[t] = float(np.dot(w, r))
        w = w * (1 + r)
        w = w / w.sum()                       # deriva natural
        if cada and (t + 1) % cada == 0:
            out[t] -= abs(w - w0).sum() * (coste / 100.0) / 2
            w = w0.copy()                     # vuelve a los pesos objetivo
    return pd.Series(out, index=ret.index)


def _pesos_paridad(ret, cols, ventana=126):
    """Pesos inversamente proporcionales a la volatilidad (paridad de riesgo simple)."""
    vol = ret[cols].rolling(ventana).std()
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.shift(1)                          # usa solo información pasada


def _cartera_dinamica(ret, w, coste=COSTE):
    cols = list(w.columns)
    # alinear estrictamente por fechas comunes: si no, los índices se desfasan
    idx = ret.index.intersection(w.index)
    R = ret.loc[idx, cols].values
    W = w.loc[idx, cols].values
    out = np.full(len(idx), np.nan)
    prev = None
    for t in range(len(idx)):
        if not (np.isfinite(R[t]).all() and np.isfinite(W[t]).all()):
            continue
        out[t] = float(np.dot(W[t], R[t]))
        if prev is not None:
            out[t] -= np.abs(W[t] - prev).sum() * (coste / 100.0) / 2
        prev = W[t]
    return pd.Series(out, index=idx)


def _fila(nombre, razon, color, est, bh):
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
                      f"<b>Sharpe {me['sharpe']} vs {mb['sharpe']}</b>"),
            "_sh": me["sharpe"], "_shb": mb["sharpe"], "_cagr": me["cagr"], "_dd": me["dd"]}


def evaluar_cartera(sintetico=False):
    px = _cargar(sintetico)
    if px is None:
        return None
    men = px.resample("ME").last()
    ret = men.pct_change().dropna(how="all")
    if "SPY" not in ret.columns or len(ret) < 80:
        return None
    spy = ret["SPY"]
    disp = [c for c in UNIVERSO if c in ret.columns]
    bloques, extras = [], []

    # --- C1: el rebalanceo como fuente de retorno ---
    base = [c for c in ("SPY", "GLD", "TLT") if c in ret.columns]
    if len(base) >= 3:
        pesos = {c: 1.0 for c in base}
        mensual = _cartera(ret, pesos, cada=1)
        anual = _cartera(ret, pesos, cada=12)
        nunca = _cartera(ret, pesos, cada=0)
        b = _fila("C1 · Rebalanceo mensual vs no rebalancear",
                  ("Rebalancear vende lo que subió y compra lo que bajó; con activos volátiles y poco "
                   "correlacionados eso genera retorno por sí solo. Se compara la MISMA cartera con "
                   "rebalanceo mensual contra dejarla derivar: la diferencia es el efecto puro."),
                  "#6ec08a", mensual, nunca)
        if b:
            bloques.append(b)
            ma, mn, mm = _met(anual.dropna().values), _met(nunca.dropna().values), _met(mensual.dropna().values)
            if ma and mn and mm:
                extras.append(
                    f"<tr><td>Cartera 1/3 bolsa, 1/3 oro, 1/3 bonos</td>"
                    f"<td>{mm['cagr']}% (Sharpe {mm['sharpe']})</td>"
                    f"<td>{ma['cagr']}% (Sharpe {ma['sharpe']})</td>"
                    f"<td>{mn['cagr']}% (Sharpe {mn['sharpe']})</td></tr>")

    # --- C2: paridad de riesgo vs pesos iguales ---
    if len(base) >= 3:
        w = _pesos_paridad(ret, base)
        par = _cartera_dinamica(ret, w.dropna())
        igual = _cartera(ret, {c: 1.0 for c in base}, cada=1)
        b = _fila("C2 · Paridad de riesgo vs pesos iguales",
                  ("Repartir para que cada activo aporte el MISMO riesgo, no el mismo capital: con "
                   "pesos iguales la bolsa domina el riesgo total y la diversificación es ilusoria."),
                  "#5fb7c4", par, igual)
        if b:
            bloques.append(b)

    # --- C3: diversificación geográfica ---
    geo = [c for c in ("SPY", "VGK", "EEM", "EWJ") if c in ret.columns]
    if len(geo) >= 3:
        mundo = _cartera(ret, {c: 1.0 for c in geo}, cada=1)
        b = _fila("C3 · Diversificación geográfica vs solo EE.UU.",
                  ("Los ciclos económicos no coinciden entre regiones, así que repartir entre EE.UU., "
                   "Europa, emergentes y Japón debería reducir riesgo sin sacrificar retorno esperado."),
                  "#b48ad6", mundo, spy)
        if b:
            bloques.append(b)

    # --- C4: bonos como activo ---
    bon = [c for c in ("TLT", "IEF") if c in ret.columns]
    if bon:
        mix = _cartera(ret, {"SPY": 0.6, bon[0]: 0.4}, cada=1)
        b = _fila("C4 · Bolsa 60 / bonos 40",
                  ("Los bonos no como señal (eso ya falló) sino como activo: su correlación negativa "
                   "con la bolsa en las crisis es lo que amortigua las caídas. La cartera clásica."),
                  "#e8b23a", mix, spy)
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

    rk = sorted(bloques, key=lambda b: -b["_sh"])
    filas_rk = "".join(
        f"<tr><td>{b['nombre'].split(' · ')[0]}</td>"
        f"<td class='{'pos' if b['_sh'] > b['_shb'] else 'est-obs'}'>{b['_sh']}</td>"
        f"<td class='est-obs'>{b['_shb']}</td><td>{b['_cagr']}%</td>"
        f"<td class='neg'>{b['_dd']}%</td></tr>" for b in rk)

    detalle = "".join(
        f"<div class='vf-row'><span class='dot' style='background:{b['color']}'></span>"
        f"<b>{b['nombre']}</b></div><div class='ch-sub' style='margin:4px 0 12px 18px'>"
        f"{b['razon']}<br><i>{b['extra']}</i></div>" for b in bloques)
    for b in bloques:
        b["tipo"] = b["nombre"]
        b["nombre"] = f"{b['nombre']} · {b['n_eventos']} meses"

    tabla_reb = ""
    if extras:
        tabla_reb = ("<br><b>El efecto del rebalanceo, aislado</b> (misma cartera, distinta frecuencia):"
                     "<div class='ops-scroll'><table class='ops'><thead><tr><th>Cartera</th>"
                     "<th>Rebalanceo mensual</th><th>Anual</th><th>Sin rebalancear</th></tr></thead>"
                     f"<tbody>{''.join(extras)}</tbody></table></div>")

    return {
        "id": "cartera",
        "etiqueta": "Construcción de cartera",
        "tipo": f"{len(bloques)} hipótesis · FDR conjunto · neto de costes · juzgado por Sharpe",
        "modelo": "cartera",
        "figuras_panel": True,
        "intro": ("Cuatro formas de construir cartera. Ninguna predice nada: solo reparten y "
                  "rebalancean. Es la única familia que ha aportado valor en este laboratorio, así "
                  "que se prueba a fondo. Mira el Sharpe y la caída, no solo el exceso." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se prueban {len(pvals)} hipótesis a la vez con corrección conjunta "
                     f"(Benjamini-Hochberg, FDR 10%): {n_fdr} sobreviven. IMPORTANTE: aquí el exceso "
                     f"de retorno NO es la métrica principal. Una cartera diversificada casi siempre "
                     f"rinde menos que la bolsa sola en un periodo alcista, y aun así puede ser "
                     f"preferible si su Sharpe es mayor y su caída mucho menor."),
        "nota": ("<b>Ranking por retorno ajustado a riesgo</b>:"
                 "<div class='ops-scroll'><table class='ops'><thead><tr><th>Hipótesis</th>"
                 "<th>Sharpe</th><th>Sharpe rival</th><th>CAGR</th><th>Caída máx.</th></tr></thead>"
                 f"<tbody>{filas_rk}</tbody></table></div>{tabla_reb}"
                 "<br>Ninguna de estas estrategias predice nada: es su virtud. No dependen de "
                 "acertar el futuro, solo de repartir riesgo y rebalancear con disciplina. "
                 "No es recomendación de inversión."),
    }
