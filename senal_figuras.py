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
LOG_COMP = "compresion_log.csv"
UMBRAL_FUERZA = 35   # solo avisa de rupturas decisivas (movimiento ≥ ~1·ATR)
TOP_COMP = 10        # nº máximo de compresiones avisadas por corrida
DIAS_DEDUP_COMP = 6  # no repetir una compresión avisada en los últimos N días


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


def _veredicto_compresion(vd_full):
    """Frase de veredicto para la compresión, leída del backtest."""
    val_sig = vd_full.get("compresion")
    if not val_sig:
        return "histórico: aún sin backtest"
    val, sig = val_sig
    if sig and val < 0:
        return f"histórico: romper aquí tendió a fallar ({val:+.1f}% a 3m)"
    if sig and val > 0:
        return f"histórico: ligera ventaja al romper ({val:+.1f}%; rara)"
    return f"histórico: romper aquí no dio ventaja fiable ({val:+.1f}%)"


def _comp_recientes():
    """Tickers de compresión avisados en los últimos DIAS_DEDUP_COMP días."""
    recientes = set()
    if not os.path.exists(LOG_COMP):
        return recientes
    limite = dt.date.today() - dt.timedelta(days=DIAS_DEDUP_COMP)
    with open(LOG_COMP, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if dt.date.fromisoformat(row["fecha"]) >= limite:
                    recientes.add(row["ticker"])
            except Exception:
                pass
    return recientes


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
    compresiones = []
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
                fz = figuras.fuerza_figura(H, L, C, i)
                if fz < UMBRAL_FUERZA:           # solo rupturas decisivas (menos ruido)
                    continue
                key = (tk, fecha, tipo)
                if key in vistas:
                    continue
                vistas.add(key)
                hallazgos.append({"ticker": tk, "fecha": fecha, "figura": tipo,
                                  "nombre": figuras.FIGURAS[tipo][0], "precio": float(C[-1]),
                                  "fuerza": int(fz)})
        # radar de compresión: estado ACTUAL del valor (a punto de romper)
        rc = figuras.radar_compresion(H, L, C)
        if rc:
            compresiones.append({"ticker": tk, "fecha": fecha, "precio": float(C[-1]), **rc})

    # registrar figuras para forward-test
    nueva = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nueva:
            w.writerow(["ticker", "fecha", "figura", "precio", "registrado"])
        for h in hallazgos:
            w.writerow([h["ticker"], h["fecha"], h["figura"], f"{h['precio']:.2f}",
                        dt.datetime.now(dt.timezone.utc).isoformat()])
    print(f"Figuras nuevas en la última sesión: {len(hallazgos)}")

    # --- mensaje 1: figuras decisivas del día ---
    if hallazgos:
        fecha = hallazgos[0]["fecha"]

        def _emoji(tipo):
            vs = vd.get(tipo)
            if vs:
                val, sig = vs
                if sig and val < 0:
                    return "🔴"
                if sig and val > 0:
                    return "🟢"
            return "⚪"

        # importancia: primero veredicto significativo, luego más fuerza
        top = sorted(hallazgos, key=lambda h: (0 if _emoji(h["figura"]) != "⚪" else 1,
                                               -h.get("fuerza", 0)))[:12]
        lineas = [f"📐 FIGURAS TÉCNICAS · {fecha}",
                  f"Las {len(top)} más decisivas de hoy (de {len(hallazgos)} detectadas). "
                  f"Informativo, no señal: las rupturas tienden a revertir.",
                  "🔴 históricamente falla · 🟢 ventaja (rara) · ⚪ sin ventaja\n"]
        for h in top:
            lineas.append(f"{_emoji(h['figura'])} {h['nombre']} · {h['ticker']} "
                          f"{h['precio']:.2f} · 💪{h.get('fuerza', 0)}")
        lineas.append("\n🌐 tristansuarez.github.io/neural-capital-research")
        lineas.append("⚠️ Educativo. No es recomendación de inversión.")
        texto = "\n".join(lineas)
        if args.telegram:
            for trozo in esc.trocear(texto):
                esc.enviar_telegram(trozo)
        else:
            print(texto)

    # --- mensaje 2: radar de compresión (las más fuertes, sin repetir) ---
    recientes = _comp_recientes()
    nuevas = [c for c in sorted(compresiones, key=lambda x: -x["fuerza"])
              if c["ticker"] not in recientes][:TOP_COMP]
    print(f"Compresiones fuertes nuevas: {len(nuevas)}")
    if nuevas:
        nuevo = not os.path.exists(LOG_COMP)
        with open(LOG_COMP, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if nuevo:
                w.writerow(["ticker", "fecha", "precio", "fuerza", "box_lo", "box_hi", "registrado"])
            for c in nuevas:
                w.writerow([c["ticker"], c["fecha"], f"{c['precio']:.2f}", int(c["fuerza"]),
                            c["box_lo"], c["box_hi"], dt.datetime.now(dt.timezone.utc).isoformat()])
        fecha = nuevas[0]["fecha"]
        lineas = [f"🎯 RADAR DE COMPRESIÓN · {fecha}",
                  "Valores muy contraídos: ruptura probable pronto. Dirección DESCONOCIDA. INFORMATIVO, no señal.",
                  f"({_veredicto_compresion(vd)})\n"]
        for c in nuevas:
            lineas.append(f"• {c['ticker']} — {c['precio']:.2f}  (compresión {int(c['fuerza'])}/100 · "
                          f"rango {c['box_lo']}–{c['box_hi']})")
        lineas.append("\n⚠️ Una compresión dice que VA a moverse, no hacia dónde ni que pague. No es recomendación.")
        texto = "\n".join(lineas)
        if args.telegram:
            for trozo in esc.trocear(texto):
                esc.enviar_telegram(trozo)
        else:
            print(texto)


if __name__ == "__main__":
    main()
