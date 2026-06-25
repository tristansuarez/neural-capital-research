#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_plata.py
==============
Señal experimental de COMPRA / VENTA de plata basada en la reversión a la media
del par oro-plata (cointegración), que es el modelo con ventaja más defendible
del laboratorio.

Idea: si el ratio oro/plata se aleja mucho de su media (la plata se pone barata
o cara respecto al oro), apostamos a que vuelve. Solo dispara cuando la
desviación es grande (|z| >= entrada). Cada señal se registra en
senal_plata_log.csv para poder medirla a futuro sin sesgo, igual que el KONCORDE.

  z alto  -> plata barata vs oro -> COMPRAR plata
  z bajo  -> plata cara  vs oro  -> VENDER plata

NO es recomendación de inversión. Es una desviación estadística, no una
predicción del precio.
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np

import config  # noqa: F401  (asegura rutas/coherencia de configuración)
import data
from models import GoldSilverPairs

LOG = "senal_plata_log.csv"
Z_ENTRY = 2.0


def calcular_senal(panel=None):
    if panel is None:
        panel = data.cargar_panel(["oro", "plata"])
    m = GoldSilverPairs(z_entry=Z_ENTRY)
    m.fit(panel)

    oro = float(panel["oro"].iloc[-1])
    plata = float(panel["plata"].iloc[-1])
    ratio = oro / plata
    fecha = panel.index[-1].strftime("%Y-%m-%d")

    if not m.cointegrated:
        return {"fecha": fecha, "plata": round(plata, 2), "oro": round(oro, 2),
                "ratio": round(ratio, 1), "z": None, "accion": "SIN SEÑAL",
                "motivo": "el par no está cointegrado ahora"}

    z = (np.log(oro) - m.beta * np.log(plata) - m.alpha - m.spread_mean) / m.spread_std
    z = float(z)
    if z >= Z_ENTRY:
        accion = "COMPRAR"
    elif z <= -Z_ENTRY:
        accion = "VENDER"
    else:
        accion = "SIN SEÑAL"
    return {"fecha": fecha, "plata": round(plata, 2), "oro": round(oro, 2),
            "ratio": round(ratio, 1), "z": round(z, 2), "accion": accion, "motivo": ""}


def registrar(s):
    """Guarda la señal en el CSV (una por fecha). Devuelve True si era nueva."""
    nuevo = not os.path.exists(LOG)
    if not nuevo:
        with open(LOG, encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] == s["fecha"]:
                    return False
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "precio_plata", "precio_oro", "ratio_oro_plata",
                        "z", "accion_plata", "registrado"])
        w.writerow([s["fecha"], s["plata"], s["oro"], s["ratio"],
                    "" if s["z"] is None else s["z"], s["accion"],
                    datetime.now().strftime("%Y-%m-%d %H:%M")])
    return True


def mensaje(s):
    if s["accion"] == "SIN SEÑAL":
        det = s["motivo"] if s["motivo"] else (f"z={s['z']:+.2f}σ" if s["z"] is not None else "")
        return (f"🥇🥈 PAR ORO-PLATA · {s['fecha']}\n"
                f"Hoy NO hay señal clara ({det}).\n"
                f"ℹ️ Sin una desviación grande, no operar es la opción honesta.\n"
                f"⚠️ Experimental y educativo. No es recomendación de inversión.")
    plata_emoji = "🟢" if s["accion"] == "COMPRAR" else "🔴"
    oro_accion = "VENDER" if s["accion"] == "COMPRAR" else "COMPRAR"
    oro_emoji = "🔴" if s["accion"] == "COMPRAR" else "🟢"
    return (f"🥇🥈 SEÑAL PAR ORO-PLATA · {s['fecha']}\n"
            f"{plata_emoji} {s['accion']} plata · 💵 {s['plata']}\n"
            f"{oro_emoji} {oro_accion} oro · 💵 {s['oro']}\n"
            f"Ratio oro/plata: {s['ratio']} · desviación z = {s['z']:+.2f}σ\n"
            f"Es un par relativo: se opera una pata contra la otra (reversión a la media).\n"
            f"ℹ️ Desviación estadística, no una predicción del precio.\n"
            f"⚠️ Experimental y educativo. No es recomendación de inversión.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true", help="Publicar en el canal.")
    ap.add_argument("--siempre", action="store_true",
                    help="Publicar también los días sin señal (por defecto se calla).")
    args = ap.parse_args()

    s = calcular_senal()
    print(f"Plata: {s['accion']}  z={s['z']}  ratio={s['ratio']}")
    if s["accion"] != "SIN SEÑAL":
        registrar(s)

    if args.telegram and (s["accion"] != "SIN SEÑAL" or args.siempre):
        from escaner_senales_telegram import enviar_telegram
        ok, det = enviar_telegram(mensaje(s))
        print(f" Telegram: {'publicado ✅' if ok else 'FALLO: ' + det}")


if __name__ == "__main__":
    main()
