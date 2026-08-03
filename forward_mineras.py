"""
forward_mineras.py — Forward-test EN VIVO de la selección de mineras.
=====================================================================
La parte del método que NO se puede backtestear (selección por salud financiera
con fundamentales actuales) se juzga aquí de la única forma limpia posible:
hacia adelante. Cada mes, el filtro se aplica con los datos de HOY, las
elegidas se escriben en forward_mineras.csv ANTES de conocer el resultado, y el
benchmark es el sector (GDX y GDXJ), no el efectivo: si la selección no bate al
ETF que no selecciona nada, el mérito era del oro.

FILTRO DE SANIDAD, PRE-REGISTRADO (no se retoca según lo que salga):
  - SENIOR (EBITDA > 0): deuda neta / EBITDA < 2  Y  flujo de caja libre > 0.
  - JUNIOR (EBITDA <= 0 o ausente): caja neta positiva (más caja que deuda) —
    quien quema caja sin colchón acaba diluyendo al accionista.
  - Baratura SOLO entre las aprobadas: seniors por EV/EBITDA ascendente,
    juniors por precio/valor contable ascendente.
  - ROBUSTEZ ANTE DATOS ROTOS (añadida tras la primera corrida real): un dato
    ausente se etiqueta "sin dato", nunca como si fuera un hecho negativo; y un
    EV/EBITDA < 0,5 se considera improbable (artefacto habitual de Yahoo con
    caja neta grande) — la empresa sigue aprobada si pasa el filtro, pero va al
    final de la cola de baratura, no en cabeza por un dato roto. El registro de
    meses anteriores NO se retoca: lo registrado, registrado queda.
  - Cartera ficticia del mes: hasta 5 seniors + hasta 3 juniors, equiponderadas.

El registro guarda también las acciones en circulación de cada elegida: con el
tiempo, el propio CSV medirá la dilución real — el dato que ningún backtest de
supervivientes puede dar. La columna `notas` queda reservada para un futuro
clasificador de noticias (el "analista" automático), que podrá anotar sin tocar
la estructura.

Universo en mineras_universo.txt (editable). Los fundamentales vienen de Yahoo
y son los VIGENTES (para un forward-test eso es exactamente lo correcto; para
un backtest sería trampa). Operaciones ficticias. NO es asesoramiento.
"""
from __future__ import annotations
import csv
import datetime as dt
import os

import numpy as np
import pandas as pd

LOG = "forward_mineras.csv"
UNIVERSO_TXT = "mineras_universo.txt"
MAX_SENIOR, MAX_JUNIOR = 5, 3
BENCH = ["GDX", "GDXJ"]


def _universo():
    if not os.path.exists(UNIVERSO_TXT):
        return []
    with open(UNIVERSO_TXT, encoding="utf-8") as fh:
        return [ln.strip().upper() for ln in fh
                if ln.strip() and not ln.strip().startswith("#")]


def _fundamentales(sintetico=False):
    """{ticker: métricas} + precios de benchmarks. Sin red en modo sintético."""
    if sintetico:
        rng = np.random.default_rng(29)
        out = {}
        for i, tk in enumerate(["SEN1", "SEN2", "SEN3", "JUN1", "JUN2", "MAL1"]):
            senior = i < 3
            ebitda = float(rng.uniform(2e8, 2e9)) if senior else float(rng.uniform(-5e7, 0))
            out[tk] = {"precio": round(float(rng.uniform(5, 60)), 2),
                       "deuda": float(rng.uniform(0, 1e9)),
                       "caja": float(rng.uniform(0, 1.5e9)),
                       "ebitda": ebitda,
                       "fcf": float(rng.uniform(-1e8, 5e8)),
                       "ev": float(rng.uniform(1e9, 2e10)),
                       "pb": round(float(rng.uniform(0.5, 6.0)), 2),
                       "acciones": int(rng.uniform(1e8, 1e9))}
        bench = {b: round(float(rng.uniform(30, 60)), 2) for b in BENCH}
        return out, bench, []
    import yfinance as yf
    out, errores = {}, []
    for tk in _universo():
        try:
            info = yf.Ticker(tk).info
            precio = info.get("regularMarketPrice") or info.get("currentPrice")
            if not precio:
                errores.append(tk); continue
            out[tk] = {"precio": round(float(precio), 2),
                       "deuda": float(info.get("totalDebt") or 0.0),
                       "caja": float(info.get("totalCash") or 0.0),
                       "ebitda": (float(info["ebitda"]) if info.get("ebitda") is not None else None),
                       "fcf": (float(info["freeCashflow"]) if info.get("freeCashflow") is not None else None),
                       "ev": (float(info["enterpriseValue"]) if info.get("enterpriseValue") else None),
                       "pb": (round(float(info["priceToBook"]), 2) if info.get("priceToBook") else None),
                       "acciones": (int(info["sharesOutstanding"]) if info.get("sharesOutstanding") else None)}
        except Exception:
            errores.append(tk)
    bench = {}
    for b in BENCH:
        try:
            s = yf.download(b, period="5d", auto_adjust=True, progress=False)["Close"].dropna()
            if len(s):
                bench[b] = round(float(s.iloc[-1].squeeze()), 2)
        except Exception:
            continue
    if not out or "GDX" not in bench:
        return None, None, errores
    return out, bench, errores


def _filtrar(datos):
    """Aplica el filtro pre-registrado. Devuelve (aprobadas ordenadas, rechazadas)."""
    seniors, juniors, rech = [], [], []
    for tk, d in datos.items():
        neta = d["deuda"] - d["caja"]
        es_senior = d["ebitda"] is not None and d["ebitda"] > 0
        if es_senior:
            apal = neta / d["ebitda"]
            fcf_ok = d["fcf"] is not None and d["fcf"] > 0
            if apal < 2.0 and fcf_ok:
                ev_eb = (d["ev"] / d["ebitda"]) if d["ev"] else None
                valido = ev_eb is not None and np.isfinite(ev_eb) and ev_eb >= 0.5
                orden = ev_eb if valido else np.inf
                met = (f"EV/EBITDA {ev_eb:.1f}" if ev_eb is not None and np.isfinite(ev_eb)
                       else "EV/EBITDA n/d")
                if not valido:
                    met += " · dato dudoso (EV improbable): relegada al final de la cola"
                seniors.append((orden, tk, d, met))
            else:
                partes = []
                if apal >= 2.0:
                    partes.append(f"deuda neta/EBITDA {apal:.1f} ≥ 2")
                if d["fcf"] is None:
                    partes.append("FCF sin dato — no verificable, no negativo")
                elif d["fcf"] <= 0:
                    partes.append("FCF ≤ 0")
                rech.append((tk, "senior", " · ".join(partes) or "no supera el filtro"))
        else:
            ok = neta < 0
            if ok:
                orden = d["pb"] if d["pb"] is not None else np.inf
                juniors.append((orden, tk, d,
                                f"caja neta {abs(neta)/1e6:.0f}M$ · P/B {d['pb']}" if d["pb"] else "caja neta positiva"))
            else:
                rech.append((tk, "junior", f"deuda neta {neta/1e6:.0f}M$ (quema sin colchón)"))
    seniors.sort(key=lambda x: x[0])
    juniors.sort(key=lambda x: x[0])
    return seniors[:MAX_SENIOR], juniors[:MAX_JUNIOR], rech


def _registrar(hoy, seniors, juniors, bench):
    mes = hoy.strftime("%Y-%m")
    ya = set()
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ya.add(row.get("mes"))
    if mes in ya:
        return 0
    nuevo = not os.path.exists(LOG)
    filas = 0
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["mes", "fecha", "ticker", "tipo", "precio", "acciones",
                        "metrica_orden", "precio_gdx", "precio_gdxj", "notas", "registrado"])
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        for _o, tk, d, met in seniors + juniors:
            tipo = "senior" if (d["ebitda"] or 0) > 0 else "junior"
            w.writerow([mes, hoy.strftime("%Y-%m-%d"), tk, tipo, d["precio"],
                        d["acciones"] or "", met, bench.get("GDX", ""),
                        bench.get("GDXJ", ""), "", ts])
            filas += 1
    return filas


def _historico():
    """Cartera equiponderada ficticia del registro vs GDX, mes a mes."""
    if not os.path.exists(LOG):
        return None
    with open(LOG, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    por_mes = {}
    for r in filas:
        por_mes.setdefault(r["mes"], []).append(r)
    meses = sorted(por_mes)
    if len(meses) < 2:
        return None
    acum_sel, acum_gdx, tramos = 1.0, 1.0, 0
    for a, b in zip(meses, meses[1:]):
        prev = {r["ticker"]: r for r in por_mes[a]}
        sig = {r["ticker"]: r for r in por_mes[b]}
        comunes = [t for t in prev if t in sig]
        # Las elegidas que desaparecen del universo al mes siguiente no se pueden
        # valorar con el propio CSV: se anota el hueco en vez de inventarlo.
        if not comunes:
            continue
        rs = []
        for t in comunes:
            try:
                rs.append(float(sig[t]["precio"]) / float(prev[t]["precio"]) - 1.0)
            except Exception:
                continue
        try:
            g = float(por_mes[b][0]["precio_gdx"]) / float(por_mes[a][0]["precio_gdx"]) - 1.0
        except Exception:
            continue
        if rs:
            acum_sel *= 1 + float(np.mean(rs))
            acum_gdx *= 1 + g
            tramos += 1
    if tramos == 0:
        return None
    return {"meses": tramos, "sel": round((acum_sel - 1) * 100, 2),
            "gdx": round((acum_gdx - 1) * 100, 2),
            "faltantes": sum(1 for a, b in zip(meses, meses[1:])
                             if not [t for t in {r['ticker'] for r in por_mes[a]}
                                     if t in {r['ticker'] for r in por_mes[b]}])}


def evaluar_forward(sintetico=False):
    datos, bench, errores = _fundamentales(sintetico)
    if not datos:
        return None
    hoy = dt.date.today()
    seniors, juniors, rech = _filtrar(datos)
    try:
        nuevas = _registrar(hoy, seniors, juniors, bench)
    except Exception:
        nuevas = 0
    hist = _historico()

    filas_sel = "".join(
        f"<tr><td><b>{tk}</b></td><td>{'senior' if (d['ebitda'] or 0) > 0 else 'junior'}</td>"
        f"<td class='mono'>{d['precio']}</td><td class='est-obs'>{met}</td></tr>"
        for _o, tk, d, met in seniors + juniors)
    filas_rech = "".join(
        f"<tr><td>{tk}</td><td>{tipo}</td><td class='est-obs'>{motivo}</td></tr>"
        for tk, tipo, motivo in rech)

    tabla_hist = ""
    # Notas del analista local (clasificador de noticias), si las hay este mes.
    notas_txt = ""
    try:
        if os.path.exists(LOG):
            mes_hoy = hoy.strftime("%Y-%m")
            with open(LOG, encoding="utf-8") as fh:
                con_notas = [r for r in csv.DictReader(fh)
                             if r.get("mes") == mes_hoy and r.get("notas")]
            if con_notas:
                filas_n = "".join(
                    f"<tr><td><b>{r['ticker']}</b></td><td class='est-obs'>{r['notas']}</td></tr>"
                    for r in con_notas)
                notas_txt = (
                    "<br><b>Notas del analista</b> (clasificador local de noticias con LLM; "
                    "prompt fijo pre-registrado en analista_mineras.py, veredictos auditables "
                    "en noticias_mineras.csv):"
                    "<div class='ops-scroll'><table class='ops'><thead><tr><th>Elegida</th>"
                    f"<th>Noticias del mes</th></tr></thead><tbody>{filas_n}</tbody></table></div>")
    except Exception:
        notas_txt = ""
    if hist:
        gana = hist["sel"] > hist["gdx"]
        tabla_hist = (
            f"<br><b>Acumulado de la selección vs GDX</b> ({hist['meses']} tramos mensuales, "
            f"operaciones ficticias verificables en forward_mineras.csv): selección "
            f"<b class='{'pos' if hist['sel'] >= 0 else 'neg'}'>{hist['sel']:+.2f}%</b> vs GDX "
            f"{hist['gdx']:+.2f}% → diferencia "
            f"<b class='{'pos' if gana else 'neg'}'>{hist['sel'] - hist['gdx']:+.2f}%</b>."
            + (f" {hist['faltantes']} tramo(s) sin tickers comunes quedaron fuera y se anotan "
               f"como hueco, no se rellenan." if hist.get("faltantes") else ""))
    else:
        tabla_hist = ("<br><div class='ch-sub'>El registro acaba de empezar: hará falta al menos "
                      "un año para que la comparación con GDX signifique algo. La espera es parte "
                      "del método.</div>")

    aviso = ""
    if errores:
        aviso = ("<br><b>Tickers sin datos en esta corrida:</b> " + ", ".join(errores) +
                 ". No se rellenan: si Yahoo no los sirve, quedan fuera del mes y se dice.")

    return {
        "id": "forward_mineras",
        "etiqueta": "Forward-test de selección de mineras (en vivo)",
        "tipo": (f"{len(seniors)} seniors + {len(juniors)} juniors aprobadas de "
                 f"{len(datos)} analizadas · registro mensual · operaciones ficticias"),
        "modelo": "forward",
        "sin_datos": True,
        "intro": (
            "<b>La parte del método de mineras que ningún backtest puede juzgar limpio</b> — la "
            "selección por salud financiera — medida de la única forma honesta: hacia adelante. "
            "Filtro pre-registrado (senior: deuda neta/EBITDA &lt; 2 y FCF &gt; 0; junior: caja "
            "neta positiva), baratura solo entre aprobadas (EV/EBITDA y P/B), y benchmark "
            "exigente: el sector entero (GDX). Si la selección no bate al ETF que no selecciona "
            "nada, el mérito era del oro."
            "<div class='ops-scroll' style='margin-top:12px'><table class='ops'><thead><tr>"
            "<th>Elegida</th><th>Tipo</th><th>Precio</th><th>Orden</th></tr></thead>"
            f"<tbody>{filas_sel}</tbody></table></div>"
            + (("<br><b>Rechazadas por el filtro</b> (el motivo es parte del registro):"
                "<div class='ops-scroll'><table class='ops'><thead><tr><th>Ticker</th><th>Tipo</th>"
                f"<th>Motivo</th></tr></thead><tbody>{filas_rech}</tbody></table></div>")
               if filas_rech else "")),
        "nota": (tabla_hist + notas_txt + aviso +
                 "<br><b>Dilución vigilada por el propio registro:</b> cada mes se guardan las "
                 "acciones en circulación de las elegidas; con el tiempo, el CSV medirá la "
                 "dilución real — el dato que el sesgo de supervivencia esconde en cualquier "
                 "backtest. La columna <code>notas</code> queda reservada para un futuro "
                 "clasificador de noticias. Fundamentales vigentes de Yahoo (correcto para un "
                 "forward; trampa para un backtest). Operaciones ficticias. No es recomendación "
                 "de inversión."),
    }
