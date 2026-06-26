#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_ventas.py — Avisos de VENTA del KONCORDE
==============================================
Para cada posición abierta del forward-test (una señal de COMPRA registrada en
`senales_log.csv` que aún no se ha vendido), comprueba si el indicador acaba de
dar señal de VENTA en la última sesión: el "marrón" cruza su media a la baja, la
regla simétrica a la compra. Si la da, avisa al canal de Telegram y lo apunta en
`ventas_log.csv` para no repetir el aviso.

Reutiliza el cálculo del indicador y el envío a Telegram del escáner de compras.

Uso:
    python senal_ventas.py            # solo detecta e imprime
    python senal_ventas.py --telegram # además publica el aviso en el canal

NO es asesoramiento financiero.
"""
from __future__ import annotations
import argparse
import csv
import os
from datetime import datetime, timedelta

import pandas as pd
import escaner_senales_telegram as esc   # reutiliza koncorde() y enviar_telegram()

BUYS  = "senales_log.csv"
SELLS = "ventas_log.csv"


def posiciones_abiertas():
    """Compras registradas que todavía no figuran como vendidas."""
    if not os.path.exists(BUYS):
        return []
    buys, vistas = [], set()
    for f in csv.DictReader(open(BUYS, encoding="utf-8")):
        clave = (f.get("ticker"), f.get("fecha"))
        if not all(clave) or clave in vistas:
            continue
        vistas.add(clave)
        buys.append({"ticker": f["ticker"], "fecha": f["fecha"], "precio": f.get("precio")})
    vendidas = set()
    if os.path.exists(SELLS):
        for f in csv.DictReader(open(SELLS, encoding="utf-8")):
            vendidas.add((f.get("ticker"), f.get("fecha_compra")))
    return [b for b in buys if (b["ticker"], b["fecha"]) not in vendidas]


def venta_en_ultima_sesion(ticker, fecha_compra):
    """Devuelve (precio_cierre, fecha) si la ÚLTIMA sesión es señal de venta
    (marrón cruza su media a la baja) y es posterior a la compra. Si no, None."""
    import yfinance as yf
    desde = (datetime.strptime(fecha_compra, "%Y-%m-%d") - timedelta(days=420)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=desde, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        return None
    if df is None or len(df) < 30:
        return None
    k = esc.koncorde(df)
    if k.index[-1] <= pd.Timestamp(fecha_compra):
        return None
    try:
        m, med = k["marron"], k["media"]
        if m.iloc[-1] < med.iloc[-1] and m.iloc[-2] >= med.iloc[-2]:
            return float(k["Close"].iloc[-1]), k.index[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def registrar_venta(ticker, fecha_compra, fecha_venta, precio):
    nuevo = not os.path.exists(SELLS)
    with open(SELLS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["ticker", "fecha_compra", "fecha_venta", "precio_venta", "registrado"])
        w.writerow([ticker, fecha_compra, fecha_venta, f"{precio:.4f}",
                    datetime.now().strftime("%Y-%m-%d %H:%M")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true", help="publicar el aviso en el canal")
    args = ap.parse_args()

    abiertas = posiciones_abiertas()
    print(f"Posiciones abiertas a revisar: {len(abiertas)}")

    ventas = []
    for p in abiertas:
        r = venta_en_ultima_sesion(p["ticker"], p["fecha"])
        if r:
            precio, fventa = r
            registrar_venta(p["ticker"], p["fecha"], fventa, precio)
            ventas.append({"ticker": p["ticker"], "fecha": fventa, "precio": precio})
            print(f"  VENTA {p['ticker']} (compra {p['fecha']}) -> {fventa} @ {precio:.2f}")

    if not ventas:
        print("Sin señales de venta hoy.")
        return

    lineas = "\n".join(f"🔴 VENDER {v['ticker']} — cierre {v['precio']:.2f}" for v in ventas)
    cuerpo = ("🔻 Señales de VENTA KONCORDE\n"
              "El indicador (marrón) ha cruzado su media a la baja en estas posiciones, "
              "que es su señal de salida:\n\n"
              f"{lineas}\n\n"
              "ℹ️ Cierra la operación que se abrió con la señal de compra.\n"
              "⚠️ Contenido educativo. No es recomendación de inversión.")
    print("\n" + cuerpo)
    if args.telegram:
        ok, detalle = esc.enviar_telegram(cuerpo)
        print(f"Telegram: {'OK' if ok else 'ERROR'} ({detalle})")


if __name__ == "__main__":
    main()
