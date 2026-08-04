#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_mineras.py — Avisos de mineras en Telegram (INFORMATIVO, no señal).
=========================================================================
Dos tipos de aviso, ambos atados a lo que el laboratorio MIDE, no a opiniones:

  1. ALTAS Y BAJAS DEL REGISTRO FORWARD. Cuando una minera entra en la selección
     del mes (pasa el filtro de sanidad y queda entre las más baratas) o sale de
     ella (deja de pasar el filtro o cae del corte), se avisa. Son los "compra"
     y "vende" FICTICIOS del registro público: el mérito o el ridículo quedarán
     escritos en forward_mineras.csv.

  2. CAMBIOS DE RÉGIMEN MACRO (las puertas de M5/M6). Cruce del oro con su media
     de 10 meses y giro del margen oro/petróleo (variación a 3 meses). Honestidad
     medida por delante: en nuestro backtest estas puertas NO batieron a GDX con
     significación; su valor demostrado fue recortar la caída máxima (−45% frente
     a −77% en M6). El aviso dice el estado, no lo que hay que hacer.

Deduplicación en senal_mineras_log.csv: cada aviso se manda una vez.
Uso:  python senal_mineras.py            (ensayo: imprime, no envía)
      python senal_mineras.py --enviar   (publica en Telegram)
NO es asesoramiento financiero.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import os

import numpy as np
import pandas as pd
import yfinance as yf

import escaner_senales_telegram as esc

LOG_AVISOS = "senal_mineras_log.csv"
LOG_FORWARD = "forward_mineras.csv"
WEB = "https://tristansuarez.github.io/neural-capital-research/lab.html?id=forward_mineras"

PIE = ("—\n📊 Registro público y verificable: " + WEB +
       "\n⚠️ Operaciones ficticias de un laboratorio. NO es recomendación de inversión.")


# ------------------------------------------------------------- avisos ----
def _ya_avisado():
    vistos = set()
    if os.path.exists(LOG_AVISOS):
        with open(LOG_AVISOS, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                vistos.add(r.get("clave"))
    return vistos


def _apuntar(claves):
    nuevo = not os.path.exists(LOG_AVISOS)
    with open(LOG_AVISOS, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["clave", "registrado"])
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        for c in claves:
            w.writerow([c, ts])


# ----------------------------------------------------- altas y bajas ----
def _seleccion_por_mes():
    """{mes: {ticker: fila}} del registro forward."""
    if not os.path.exists(LOG_FORWARD):
        return {}
    out = {}
    with open(LOG_FORWARD, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["mes"], {})[r["ticker"]] = r
    return out


def avisos_seleccion(vistos):
    sel = _seleccion_por_mes()
    meses = sorted(sel)
    if not meses:
        return [], []
    lineas, claves = [], []
    ultimo = meses[-1]
    ahora = sel[ultimo]
    previo = sel[meses[-2]] if len(meses) >= 2 else {}

    if len(meses) < 2:
        # primer mes del registro: presentar la selección inicial una sola vez
        clave = f"inicial:{ultimo}"
        if clave not in vistos:
            lineas.append(f"🟡 Registro inaugural {ultimo} — selección inicial del filtro:")
            for tk, r in sorted(ahora.items()):
                nota = f" · {r['notas']}" if r.get("notas") else ""
                lineas.append(f"   • {tk} ({r['tipo']}) a {r['precio']} $ · {r['metrica_orden']}{nota}")
            claves.append(clave)
        return lineas, claves

    altas = sorted(set(ahora) - set(previo))
    bajas = sorted(set(previo) - set(ahora))
    for tk in altas:
        clave = f"alta:{tk}:{ultimo}"
        if clave in vistos:
            continue
        r = ahora[tk]
        nota = f" · noticias: {r['notas']}" if r.get("notas") else ""
        lineas.append(f"🟢 ALTA {tk} ({r['tipo']}) — entra en la selección de {ultimo} "
                      f"a {r['precio']} $ · {r['metrica_orden']}{nota}")
        claves.append(clave)
    for tk in bajas:
        clave = f"baja:{tk}:{ultimo}"
        if clave in vistos:
            continue
        r = previo[tk]
        lineas.append(f"🔴 BAJA {tk} — sale de la selección en {ultimo} "
                      f"(entró a {r['precio']} $ en {r['mes']}). El motivo queda en la web.")
        claves.append(clave)
    if not altas and not bajas:
        clave = f"sin_cambios:{ultimo}"
        if clave not in vistos:
            lineas.append(f"⚪ Selección de {ultimo} sin cambios: " + ", ".join(sorted(ahora)))
            claves.append(clave)
    return lineas, claves


# ------------------------------------------------------------- macro ----
def _estado_macro(sintetico=False):
    """(oro_sobre_ma10, cruce_oro, margen_sube, giro_margen) con datos mensuales."""
    if sintetico:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2015-01-01", periods=2400)
        gld = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.009, len(idx)))), idx)
        cl = pd.Series(100 * np.exp(np.cumsum(rng.normal(1e-4, 0.015, len(idx)))), idx)
    else:
        try:
            gld = yf.download("GLD", period="12y", auto_adjust=True,
                              progress=False)["Close"].dropna().squeeze()
            cl = yf.download("CL=F", period="12y", auto_adjust=True,
                             progress=False)["Close"].dropna().squeeze()
        except Exception:
            return None
        if len(gld) < 500:
            return None
    men_g = gld.resample("ME").last().dropna()
    ma10 = men_g.rolling(10).mean()
    sobre = bool(men_g.iloc[-1] > ma10.iloc[-1])
    sobre_prev = bool(men_g.iloc[-2] > ma10.iloc[-2]) if len(men_g) > 11 else sobre
    cruce = (sobre != sobre_prev)
    margen_sube, giro = None, False
    try:
        men_c = cl.resample("ME").last().dropna()
        m = (men_g / men_c.reindex(men_g.index)).dropna()
        var3 = m.pct_change(3)
        margen_sube = bool(var3.iloc[-1] > 0)
        giro = bool((var3.iloc[-2] > 0) != margen_sube) if len(var3) > 4 else False
    except Exception:
        pass
    return {"mes": men_g.index[-1].strftime("%Y-%m"), "sobre": sobre, "cruce": cruce,
            "margen_sube": margen_sube, "giro": giro}


def avisos_macro(vistos, sintetico=False):
    e = _estado_macro(sintetico)
    if e is None:
        return [], []
    lineas, claves = [], []
    if e["cruce"]:
        clave = f"cruce_oro:{e['mes']}:{'arriba' if e['sobre'] else 'abajo'}"
        if clave not in vistos:
            if e["sobre"]:
                lineas.append("🟢 RÉGIMEN: el oro cruza SOBRE su media de 10 meses. "
                              "En el backtest (M5/M6), estar en mineras solo en este régimen no dio "
                              "más retorno que GDX, pero recortó la caída máxima de −77% a −45%.")
            else:
                lineas.append("🔴 RÉGIMEN: el oro cruza BAJO su media de 10 meses. "
                              "El régimen que históricamente concentró los desastres del sector: "
                              "en el backtest, salir aquí fue lo que evitó el −77%.")
            claves.append(clave)
    if e["giro"] and e["margen_sube"] is not None:
        clave = f"giro_margen:{e['mes']}:{'sube' if e['margen_sube'] else 'baja'}"
        if clave not in vistos:
            lineas.append(("🟢 Margen oro/petróleo girando al alza (variación 3m positiva). "
                           if e["margen_sube"] else
                           "🔴 Margen oro/petróleo girando a la baja (variación 3m negativa). ")
                          + "En el backtest esta señal sola fue ruido (M4, p=0,58): es contexto, no edge.")
            claves.append(clave)
    return lineas, claves


# ---------------------------------------------------------------- main ----
def main(enviar=False, sintetico=False):
    vistos = _ya_avisado()
    l1, c1 = avisos_seleccion(vistos)
    l2, c2 = avisos_macro(vistos, sintetico)
    lineas, claves = l1 + l2, c1 + c2
    if not lineas:
        print("Sin avisos nuevos.")
        return
    cabecera = "⛏️ MINERAS · Neural Capital Research\n"
    cuerpo = cabecera + "\n".join(lineas)
    if enviar:
        ok_alguno = False
        for trozo in esc.trocear_con_pie(cuerpo, PIE):
            ok, msg = esc.enviar_telegram(trozo)
            ok_alguno = ok_alguno or ok
            if not ok:
                print("Telegram:", msg)
        if ok_alguno:
            _apuntar(claves)
            print(f"Enviados {len(lineas)} aviso(s); {len(claves)} clave(s) apuntadas.")
        else:
            print("No se envió nada: no se apunta la deduplicación (se reintentará).")
    else:
        print("--- ENSAYO (no se envía; usa --enviar) ---")
        print(cuerpo.replace("", "").replace("", ""))
        print(PIE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enviar", action="store_true")
    ap.add_argument("--sintetico", action="store_true")
    args = ap.parse_args()
    main(enviar=args.enviar, sintetico=args.sintetico)
