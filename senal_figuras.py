#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_figuras.py — Avisos de figuras tecnicas (INFORMATIVO, no senal).
======================================================================
Escanea el S&P 500, detecta figuras formadas en la ULTIMA sesion (reglas fijas de
figuras.py) y publica en Telegram cada una con su VEREDICTO historico, leido del
backtest de figuras que vive en resultados.json.

Honestidad por delante: en nuestros propios datos las rupturas tienden a REVERTIR,
no a continuar. Esto es contexto medido, NO una recomendacion de compra/venta.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

import figuras
import escaner_senales_telegram as esc

LOG = "figuras_log.csv"


def _veredictos():
    """Lee resultados.json: {tipo_figura: (ventaja_largo_plazo, sobrevive_fdr)}."""
    try:
        with open("resultados.json", encoding="utf-8") as fh:
            d = json.load(fh)
        fig = next(e for e in d["experimentos"] if e.get("figuras_panel"))
    except Exception:
        return {}
    out = {}
    for f in fig.get("figuras", []):
        p = next((p for p in f["puntos"] if p["etiqueta"] == "3 meses"), None) \
            or (f["puntos"][-1] if f["puntos"] else None)
        if p:
            out[f["tipo"]] = (p["valor"], bool(p.get("sig_fdr", False)))
    return out


def _frase_veredicto(tipo, vd):
    if tipo not in vd:
        return "histórico: aún sin backtest"
    val, sig = vd[tipo]
    if sig and val < 0:
        return f"histórico 3m: {val:+.1f}% — tiende a FALLAR (el precio suele ir al contrario)"
    if sig and val > 0:
        return f"histórico 3m: {val:+.1f}% — ventaja a favor (rara; con cautela)"
    return f"histórico 3m: {val:+.1f}% — sin ventaja fiable (abraza el cero)"


def _ya_registradas():
    vistas = set()
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vistas.add((row["ticker"], row["fecha"], row["figura"]))
    return vistas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    args = ap.parse_args()

    vd = _veredictos()
    tickers = args.tickers or esc.obtener_sp500()
    inicio = (dt.date.today() - dt.timedelta(days=420)).isoformat()
    vistas = _ya_registradas()

    hallazgos = []
    for tk in tickers:
        try:
            df = yf.download(tk, start=inicio, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 120:
                continue
            H = np.asarray(df["High"].values, float)
            L = np.asarray(df["Low"].values, float)
            C = np.asarray(df["Close"].values, float)
            fecha = df.index[-1].strftime("%Y-%m-%d")
            ev = figuras.detectar(H, L, C)
        except Exception:
            continue
        n = len(C)
        for (i, tipo, _d) in ev:
            if i >= n - 1:                       # figura formada en la última sesión
                key = (tk, fecha, tipo)
                if key in vistas:
                    continue
                vistas.add(key)
                hallazgos.append({"ticker": tk, "fecha": fecha, "figura": tipo,
                                  "nombre": figuras.FIGURAS[tipo][0], "precio": float(C[-1])})

    # registrar para forward-test
    nueva = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nueva:
            w.writerow(["ticker", "fecha", "figura", "precio", "registrado"])
        for h in hallazgos:
            w.writerow([h["ticker"], h["fecha"], h["figura"], f"{h['precio']:.2f}",
                        dt.datetime.now(dt.timezone.utc).isoformat()])

    print(f"Figuras nuevas en la última sesión: {len(hallazgos)}")
    if not hallazgos:
        return

    fecha = hallazgos[0]["fecha"]
    lineas = [f"📐 FIGURAS TÉCNICAS · {fecha}",
              "Detectadas con reglas fijas en el cierre. INFORMATIVO, no señal: en nuestros datos "
              "las rupturas tienden a REVERTIR, no a continuar.\n"]
    for h in sorted(hallazgos, key=lambda x: x["nombre"]):
        lineas.append(f"• {h['nombre']} en {h['ticker']} — {h['precio']:.2f}")
        lineas.append(f"   {_frase_veredicto(h['figura'], vd)}")
    lineas.append("\n🌐 tristansuarez.github.io/neural-capital-research")
    lineas.append("⚠️ Contenido educativo. No es recomendación de inversión.")
    texto = "\n".join(lineas)

    if args.telegram:
        for trozo in esc.trocear(texto):
            esc.enviar_telegram(trozo)
    else:
        print(texto)


if __name__ == "__main__":
    main()
