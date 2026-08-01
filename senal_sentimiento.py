#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_sentimiento.py — Aviso de sentimiento extremo del mercado (VIX contrario).
================================================================================
Comprueba si el VIX está HOY en un extremo (decil alto = miedo, decil bajo =
euforia) respecto a sus últimos ~2 años, y avisa por Telegram con la lectura
contraria y el veredicto del event-study (resultados.json -> sentimiento_vix).

Es el registro EN VIVO de la hipótesis contraria. Informativo, NO señal.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os

import sentimiento
import escaner_senales_telegram as esc

LOG = "sentimiento_log.csv"


def _veredicto():
    try:
        with open("resultados.json", encoding="utf-8") as fh:
            d = json.load(fh)
        e = next(x for x in d["experimentos"] if x.get("id") == "sentimiento_vix")
    except Exception:
        return {}
    out = {}
    for f in e.get("figuras", []):
        p = next((p for p in f["puntos"] if p["etiqueta"] == "1 mes"), None) \
            or (f["puntos"][-1] if f["puntos"] else None)
        if p:
            out[f["tipo"]] = (p["valor"], bool(p.get("sig_fdr", False)))
    return out


def _frase(senal, vd):
    v = vd.get(senal)
    if not v:
        return "histórico: aún sin backtest"
    val, sig = v
    if sig and val > 0:
        return f"histórico (1 mes): la hipótesis SÍ aguantó ({val:+.1f}% sobre la media)"
    return f"histórico (1 mes): sin ventaja fiable ({val:+.1f}%)"


def _ya_avisado_hoy(fecha):
    if not os.path.exists(LOG):
        return False
    with open(LOG, encoding="utf-8") as fh:
        return any(row.get("fecha") == fecha for row in csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true")
    args = ap.parse_args()

    est = sentimiento.estado_actual()
    if est is None:
        print("Sin datos de VIX.")
        return
    print("Estado:", est)
    if est["senal"] is None:
        return  # zona normal, no se avisa (el silencio es información)

    fecha = dt.date.today().isoformat()
    if _ya_avisado_hoy(fecha):
        print("Ya avisado hoy.")
        return

    vd = _veredicto()
    emoji = "😱" if est["senal"] == "miedo" else "🤑"
    cuerpo = [f"{emoji} SENTIMIENTO EXTREMO · {fecha}",
              f"VIX en {est['vix']} (percentil {est['percentil']} de los últimos 2 años).",
              f"{est['lectura']}.",
              _frase(est["senal"], vd)]
    pie = ("🌐 tristansuarez.github.io/neural-capital-research\n"
           "⚠️ Hipótesis contraria en prueba. No es recomendación de inversión.")
    texto = "\n".join(cuerpo) + "\n\n" + pie

    nuevo = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["fecha", "senal", "vix", "percentil", "registrado"])
        w.writerow([fecha, est["senal"], est["vix"], est["percentil"],
                    dt.datetime.now(dt.timezone.utc).isoformat()])

    if args.telegram:
        esc.enviar_telegram(texto)
    else:
        print(texto)


if __name__ == "__main__":
    main()
