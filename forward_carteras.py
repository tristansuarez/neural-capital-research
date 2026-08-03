"""
forward_carteras.py — Forward-test EN VIVO de las estrategias que han sobrevivido.
=================================================================================
De 27 experimentos, tres han quedado como aprovechables (no baten al mercado en
rentabilidad, pero dan casi el mismo retorno con mucho menos riesgo):

  V1  Volatilidad objetivo (12% anual). Escala la exposición al S&P según la
      volatilidad realizada: menos cuando hay tormenta, más cuando hay calma.
      Sharpe 0,91 vs 0,77 y caída -28% vs -51% en el backtest.
  B3  Defensivos + oro al romperse la tendencia.
  C4  Bolsa 60 / bonos 40, la cartera clásica.

Este módulo NO es el backtest: es el REGISTRO EN VIVO. Cada corrida calcula la
posición que toca hoy y la guarda en forward_carteras.csv. Con el tiempo, el
histórico acumulado dirá si lo que funcionó en el pasado sigue funcionando —que es
la única prueba que de verdad importa y la que casi nadie publica.

Las operaciones son FICTICIAS: es un registro público y verificable, no una cartera
real. NO es asesoramiento financiero.
"""
from __future__ import annotations
import csv
import datetime as dt
import os

import numpy as np
import pandas as pd

LOG = "forward_carteras.csv"
VOL_OBJETIVO = 0.12
ACTIVOS = ["SPY", "GLD", "TLT", "XLU", "XLP"]


def _cargar(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(111)
        n = 900
        idx = pd.bdate_range("2023-01-01", periods=n)
        return pd.DataFrame({tk: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
                             for tk in ACTIVOS}, index=idx)
    import yfinance as yf
    try:
        df = yf.download(ACTIVOS, period="3y", auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
    except Exception:
        return None
    out = {}
    for tk in ACTIVOS:
        try:
            s = df[tk]["Close"].dropna()
            if len(s) > 300:
                out[tk] = s
        except Exception:
            continue
    return pd.DataFrame(out).ffill() if "SPY" in out else None


def _posiciones(px):
    """Calcula la posición que toca HOY para cada estrategia, con datos hasta hoy."""
    hoy = px.index[-1]
    dia = px.pct_change()
    pos = {}

    # V1: exposición = objetivo / volatilidad realizada (21 días)
    vol = float(dia["SPY"].tail(21).std() * np.sqrt(252))
    expo = float(np.clip(VOL_OBJETIVO / vol, 0.3, 2.0)) if vol > 0 else 1.0
    pos["V1"] = {"nombre": "Volatilidad objetivo (12%)",
                 "detalle": f"SPY al {expo * 100:.0f}% · resto en liquidez",
                 "expo": round(expo, 2),
                 "contexto": f"volatilidad actual {vol * 100:.1f}% anual"}

    # B3: si el SPY está sobre su media de 10 meses -> bolsa; si no -> defensivos + oro
    men = px["SPY"].resample("ME").last()
    ma10 = float(men.tail(10).mean()) if len(men) >= 10 else float(men.mean())
    alcista = float(men.iloc[-1]) > ma10
    pos["B3"] = {"nombre": "Defensivos + oro al romperse la tendencia",
                 "detalle": ("SPY al 100%" if alcista else "50% defensivos (XLU/XLP) + 50% oro (GLD)"),
                 "expo": 1.0,
                 "contexto": f"SPY {'por encima' if alcista else 'por debajo'} de su media de 10 meses"}

    # C4: 60/40 fijo
    pos["C4"] = {"nombre": "Bolsa 60 / bonos 40",
                 "detalle": "60% SPY + 40% TLT, rebalanceo mensual",
                 "expo": 1.0,
                 "contexto": "posición fija, sin señales"}
    return hoy, pos


def _registrar(hoy, pos, px):
    """Guarda la posición del mes si aún no está registrada."""
    mes = hoy.strftime("%Y-%m")
    ya = set()
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ya.add((row.get("mes"), row.get("estrategia")))
    nuevo = not os.path.exists(LOG)
    filas = 0
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["mes", "fecha", "estrategia", "posicion", "exposicion",
                        "precio_spy", "registrado"])
        for k, v in pos.items():
            if (mes, k) in ya:
                continue
            w.writerow([mes, hoy.strftime("%Y-%m-%d"), k, v["detalle"], v["expo"],
                        round(float(px["SPY"].iloc[-1]), 2),
                        dt.datetime.now(dt.timezone.utc).isoformat()])
            filas += 1
    return filas


def _historico():
    """Lee el registro y calcula el resultado acumulado de cada estrategia."""
    if not os.path.exists(LOG):
        return []
    filas = []
    with open(LOG, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    if len(filas) < 2:
        return []
    por_est = {}
    for r in filas:
        por_est.setdefault(r["estrategia"], []).append(r)
    out = []
    for est, rs in por_est.items():
        rs = sorted(rs, key=lambda x: x["mes"])
        if len(rs) < 2:
            continue
        ret_est, ret_bh = 1.0, 1.0
        for a, b in zip(rs, rs[1:]):
            try:
                p0, p1 = float(a["precio_spy"]), float(b["precio_spy"])
                e = float(a["exposicion"])
            except Exception:
                continue
            r_mercado = p1 / p0 - 1.0
            ret_est *= (1 + r_mercado * e)
            ret_bh *= (1 + r_mercado)
        out.append({"estrategia": est, "meses": len(rs) - 1,
                    "acum": round((ret_est - 1) * 100, 2),
                    "acum_bh": round((ret_bh - 1) * 100, 2)})
    return out


def evaluar_forward(sintetico=False):
    px = _cargar(sintetico)
    if px is None:
        return None
    hoy, pos = _posiciones(px)
    try:
        nuevas = _registrar(hoy, pos, px)
    except Exception:
        nuevas = 0
    hist = _historico()

    filas_pos = "".join(
        f"<tr><td><b>{v['nombre']}</b></td><td>{v['detalle']}</td>"
        f"<td class='est-obs'>{v['contexto']}</td></tr>" for v in pos.values())

    filas_hist = "".join(
        f"<tr><td>{h['estrategia']}</td><td class='mono'>{h['meses']}</td>"
        f"<td class='{'pos' if h['acum'] >= 0 else 'neg'}'>{h['acum']:+.2f}%</td>"
        f"<td class='est-obs'>{h['acum_bh']:+.2f}%</td>"
        f"<td class='{'pos' if h['acum'] > h['acum_bh'] else 'est-obs'}'>"
        f"{h['acum'] - h['acum_bh']:+.2f}%</td></tr>" for h in hist)

    tabla_hist = ""
    if filas_hist:
        tabla_hist = ("<br><b>Resultado acumulado desde que empezó el registro</b> "
                      "(operaciones ficticias, verificables en forward_carteras.csv):"
                      "<div class='ops-scroll'><table class='ops'><thead><tr><th>Estrategia</th>"
                      "<th>Meses</th><th>Acumulado</th><th>Comprar y mantener</th><th>Diferencia</th>"
                      f"</tr></thead><tbody>{filas_hist}</tbody></table></div>")
    else:
        tabla_hist = ("<br><div class='ch-sub'>Aún no hay histórico suficiente: el registro acaba de "
                      "empezar. Hará falta al menos un año para que los números signifiquen algo, y "
                      "varios para poder afirmar nada. Esa espera es parte del método.</div>")

    return {
        "id": "forward_carteras",
        "etiqueta": "Forward-test de carteras (en vivo)",
        "tipo": f"3 estrategias · registro mensual desde {hoy.strftime('%Y-%m')} · operaciones ficticias",
        "modelo": "forward",
        "sin_datos": True,
        "intro": ("<b>Posición que toca este mes</b> según cada estrategia superviviente del "
                  "laboratorio. Se registra automáticamente cada corrida y se compara con comprar y "
                  "mantener. Es la única prueba que importa de verdad: si lo que funcionó en el "
                  "pasado sigue funcionando."
                  "<div class='ops-scroll' style='margin-top:12px'><table class='ops'><thead><tr>"
                  "<th>Estrategia</th><th>Posición hoy</th><th>Por qué</th></tr></thead>"
                  f"<tbody>{filas_pos}</tbody></table></div>"),
        "nota": (tabla_hist +
                 "<br><b>Por qué esto importa:</b> las tres estrategias vienen de un backtest, y un "
                 "backtest siempre favorece a quien lo hace. Este registro se escribe ANTES de "
                 "conocer el resultado, así que no se puede retocar a posteriori. Si dentro de unos "
                 "años estas estrategias siguen dando lo que prometían —casi el mismo retorno con la "
                 "mitad del riesgo—, será una afirmación con fundamento. Si no, quedará registrado "
                 "igualmente.<br><br>Ninguna de las tres bate al mercado en rentabilidad: su valor "
                 "está en el riesgo. Operaciones ficticias. No es recomendación de inversión."),
    }
