"""
momentum_tsm.py — Estudio a fondo del momentum de serie temporal (H4).
=====================================================================
H4 fue la única hipótesis del laboratorio que acertó lo que prometía: no más
retorno, sino MENOS CAÍDA. Aquí se estudia en serio, no para buscar significancia
sino para entender el efecto:

  1. ¿Aguanta con distintas medias (6, 8, 10, 12 meses)? Si solo funciona con una,
     es ajuste de parámetro; si funciona con todas, es un efecto robusto.
  2. ¿Funciona en varios activos (S&P 500, oro, plata, cobre)? Si es real, debería
     aparecer en mercados distintos.
  3. ¿Cuánto cuesta la protección? Se mide el retorno sacrificado frente al
     drawdown evitado, y cuántos meses se pasa fuera del mercado.
  4. ¿Y en el peor caso? Se reporta el periodo en que la estrategia MÁS perdió
     frente a comprar y mantener (el coste de estar fuera en rebotes fuertes).

Lo que se juzga es Sharpe y caída máxima, NO exceso de retorno: la hipótesis nunca
prometió ganar más. NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import data
import config

MEDIAS = [6, 8, 10, 12]
COSTE_MES = 0.05


def _serie_mensual(px):
    return px.resample("ME").last()


def _metricas(ret, dentro=None, coste=COSTE_MES):
    """Sharpe, CAGR y caída máxima de una serie de retornos mensuales."""
    r = np.asarray(ret, float)
    r = r[np.isfinite(r)]
    if len(r) < 24:
        return None
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
    sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(12)) if np.std(r) > 0 else 0.0
    cagr = float((eq[-1] ** (12 / len(r)) - 1) * 100)
    return {"sharpe": round(sh, 2), "cagr": round(cagr, 1), "dd": round(dd, 1), "n": len(r)}


def _tsm(men, ventana, coste=COSTE_MES):
    """Estrategia: dentro si el precio del mes anterior > su media de N meses."""
    ret = men.pct_change()
    ma = men.rolling(ventana).mean()
    dentro = (men.shift(1) > ma.shift(1))
    cambios = dentro.astype(float).diff().abs().fillna(0)
    est = ret.where(dentro, 0.0) - cambios * (coste / 100.0)
    df = pd.DataFrame({"est": est, "bh": ret}).dropna()
    if len(df) < 36:
        return None
    m_est = _metricas(df["est"].values)
    m_bh = _metricas(df["bh"].values)
    if not m_est or not m_bh:
        return None
    fuera = float((~dentro.reindex(df.index).fillna(False)).mean() * 100)
    # peor tramo relativo: 12 meses donde más se quedó atrás
    rel = (df["est"] - df["bh"]).rolling(12).sum() * 100
    peor = float(rel.min()) if rel.notna().any() else float("nan")
    peor_f = rel.idxmin().strftime("%Y-%m") if rel.notna().any() else "—"
    return {"ventana": ventana, "est": m_est, "bh": m_bh, "fuera_pct": round(fuera),
            "peor_12m": round(peor, 1), "peor_fecha": peor_f}


def evaluar_tsm(sintetico=False):
    activos = []
    if sintetico:
        syn = data.cargar_sinteticos()
        for c in syn.columns:
            activos.append((c, syn[c]))
    else:
        for nombre in ("sp500", "oro", "plata", "cobre"):
            try:
                s = data.cargar_panel([nombre])[nombre]
                if len(s) > 1500:
                    activos.append((nombre, s))
            except Exception:
                continue
    if not activos:
        return None

    filas, bloques = [], []
    for nombre, px in activos:
        men = _serie_mensual(px)
        res = []
        for v in MEDIAS:
            r = _tsm(men, v)
            if r:
                res.append(r)
        if not res:
            continue
        bloques.append((nombre, res))
        for r in res:
            filas.append(
                f"<tr><td>{nombre}</td><td>{r['ventana']}m</td>"
                f"<td class='{'pos' if r['est']['sharpe'] >= r['bh']['sharpe'] else 'neg'}'>"
                f"{r['est']['sharpe']:.2f}</td><td class='est-obs'>{r['bh']['sharpe']:.2f}</td>"
                f"<td class='{'pos' if r['est']['dd'] > r['bh']['dd'] else 'neg'}'>{r['est']['dd']:.0f}%</td>"
                f"<td class='est-obs'>{r['bh']['dd']:.0f}%</td>"
                f"<td>{r['est']['cagr']:.1f}%</td><td class='est-obs'>{r['bh']['cagr']:.1f}%</td>"
                f"<td class='est-obs'>{r['fuera_pct']}%</td>"
                f"<td class='neg'>{r['peor_12m']:.0f}%</td></tr>")
    if not bloques:
        return None

    # ¿en cuántas combinaciones mejora Sharpe y reduce caída?
    tot = sum(len(r) for _n, r in bloques)
    mejor_sh = sum(1 for _n, rs in bloques for r in rs if r["est"]["sharpe"] > r["bh"]["sharpe"])
    mejor_dd = sum(1 for _n, rs in bloques for r in rs if r["est"]["dd"] > r["bh"]["dd"])
    coste_ret = float(np.mean([r["est"]["cagr"] - r["bh"]["cagr"] for _n, rs in bloques for r in rs]))
    ahorro_dd = float(np.mean([r["est"]["dd"] - r["bh"]["dd"] for _n, rs in bloques for r in rs]))

    # curva de la mejor documentada (S&P, media 10m) para el gráfico
    curva, curva2, titulo = [], [], "Momentum de serie temporal"
    for nombre, px in activos:
        if nombre in ("sp500", "oro"):
            men = _serie_mensual(px)
            ret = men.pct_change()
            ma = men.rolling(10).mean()
            dentro = (men.shift(1) > ma.shift(1))
            cambios = dentro.astype(float).diff().abs().fillna(0)
            est = (ret.where(dentro, 0.0) - cambios * (COSTE_MES / 100.0)).dropna()
            bh = ret.reindex(est.index)
            eq_e = np.cumprod(1 + est.values); eq_b = np.cumprod(1 + bh.values)
            fechas = [d.strftime("%Y-%m-%d") for d in est.index]
            curva = [{"fecha": f, "valor": round(float(v), 3)} for f, v in zip(fechas, eq_e)]
            curva2 = [{"fecha": f, "valor": round(float(v), 3)} for f, v in zip(fechas, eq_b)]
            titulo = f"Momentum de serie temporal · {nombre} · media de 10 meses"
            break

    return {
        "id": "momentum_tsm",
        "etiqueta": "Momentum de serie temporal (protección)",
        "tipo": f"{len(bloques)} activos × {len(MEDIAS)} medias · juzgado por riesgo, no por retorno",
        "modelo": "tsm",
        "color": "#6ec08a",
        "headline": {"valor": round(ahorro_dd, 1),
                     "etiqueta": "Reducción media de la caída máxima frente a comprar y mantener",
                     "sufijo": " pts", "decimales": 1},
        "significancia": {"p_valor": None, "ic90": None,
                          "etiqueta": "se juzga por riesgo (Sharpe y caída), no por significancia de retorno"},
        "cards": [
            {"k": "Mejora el Sharpe en", "v": f"{mejor_sh} de {tot} combinaciones", "tono": ""},
            {"k": "Reduce la caída máxima en", "v": f"{mejor_dd} de {tot} combinaciones", "tono": ""},
            {"k": "Coste medio en rentabilidad anual", "v": f"{coste_ret:+.1f} pts", "tono": ""},
            {"k": "Ahorro medio en caída máxima", "v": f"{ahorro_dd:+.1f} pts", "tono": ""},
        ],
        "diagnostico": {},
        "curva": curva,
        "curva2": {"nombre": "Comprar y mantener", "datos": curva2},
        "curva_color": "#6ec08a",
        "curva_unidad": "×",
        "curva_base": 1.0,
        "curva_titulo": titulo,
        "curva_sub": ("La estrategia sale del mercado cuando el precio cae bajo su media. Suele quedar "
                      "por debajo en las subidas largas y evitar el grueso de los desplomes. Su valor "
                      "no es ganar más, sino caer menos."),
        "nota": ("<b>Detalle por activo y ventana</b> (verde = mejor que comprar y mantener):"
                 "<div class='ops-scroll'><table class='ops'><thead><tr>"
                 "<th>Activo</th><th>Media</th><th>Sharpe est.</th><th>Sharpe B&H</th>"
                 "<th>Caída est.</th><th>Caída B&H</th><th>CAGR est.</th><th>CAGR B&H</th>"
                 "<th>% fuera</th><th>Peor 12m rel.</th>"
                 f"</tr></thead><tbody>{''.join(filas)}</tbody></table></div>"
                 "<br>La columna «Peor 12m rel.» es lo máximo que la estrategia se quedó atrás en un "
                 "año: ese es el precio psicológico de la protección, ver al mercado subir estando "
                 "fuera. Si el efecto solo apareciera con una media concreta, sería ajuste de "
                 "parámetro; que aparezca con varias y en varios activos es señal de robustez. "
                 "Coste de 0,05% por cambio de posición. No es recomendación de inversión."),
    }
