#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_figuras_mensual.py — Avisos de figuras en velas MENSUALES (INFORMATIVO).
=============================================================================
Corre una vez al mes. Detecta figuras cerradas en la última vela mensual COMPLETA
(descarta el mes en curso) y las publica en Telegram con su veredicto del backtest
mensual (resultados.json -> figuras_mensual).

SEND-ONLY: no commitea nada (no dispara el laboratorio). Honestamente, las figuras
mensuales son escasas, lentísimas y llegan muy a toro pasado; esto es contexto de
fondo, NO una señal.
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

import figuras
import escaner_senales_telegram as esc

UNIVERSO = 200        # se escanea 1 vez al mes; podemos abarcar más valores
UMBRAL_FUERZA = 30    # umbral algo más bajo: en mensual todo es de mayor tamaño


def _veredictos():
    try:
        with open("resultados.json", encoding="utf-8") as fh:
            d = json.load(fh)
        fig = next(e for e in d["experimentos"] if e.get("id") == "figuras_mensual")
    except Exception:
        return {}
    out = {}
    for f in fig.get("figuras", []):
        p = next((p for p in f["puntos"] if p["etiqueta"] == "3 meses"), None) \
            or (f["puntos"][-1] if f["puntos"] else None)
        if p:
            out[f["tipo"]] = (p["valor"], bool(p.get("sig_fdr", False)))
    return out


def _emoji(tipo, vd):
    vs = vd.get(tipo)
    if vs:
        val, sig = vs
        if sig and val < 0:
            return "🔴"
        if sig and val > 0:
            return "🟢"
    return "⚪"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    args = ap.parse_args()

    vd = _veredictos()
    tickers = args.tickers or esc.obtener_sp500()[:UNIVERSO]
    mes_actual = dt.date.today().strftime("%Y-%m")
    hall = []
    for j in range(0, len(tickers), 40):
        chunk = tickers[j:j + 40]
        try:
            df = yf.download(chunk, period="15y", interval="1mo", group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for tk in chunk:
            try:
                sub = df[tk].dropna()
                if len(sub) < 40:
                    continue
                H = np.asarray(sub["High"].values, float)
                L = np.asarray(sub["Low"].values, float)
                C = np.asarray(sub["Close"].values, float)
                fechas = [d.strftime("%Y-%m") for d in sub.index]
                # última vela COMPLETA: descarta el mes en curso
                ult = len(C) - 1
                if fechas[-1] == mes_actual and len(C) > 1:
                    ult = len(C) - 2
                ev = figuras.detectar(H[:ult + 1], L[:ult + 1], C[:ult + 1])
            except Exception:
                continue
            for (i, tipo, _d) in ev:
                if i >= ult:
                    fz = figuras.fuerza_figura(H, L, C, ult)
                    if fz < UMBRAL_FUERZA:
                        continue
                    hall.append({"ticker": tk, "mes": fechas[ult], "figura": tipo,
                                 "nombre": figuras.FIGURAS[tipo][0],
                                 "precio": float(C[ult]), "fuerza": int(fz)})

    print(f"Figuras mensuales cerradas: {len(hall)}")
    if not hall:
        return

    top = sorted(hall, key=lambda h: (0 if _emoji(h["figura"], vd) != "⚪" else 1,
                                      -h["fuerza"]))
    mes = top[0]["mes"]
    cuerpo = [f"📅 FIGURAS MENSUALES · {mes}",
              f"Figuras cerradas en la vela mensual ({len(top)}). Informativo, no señal.",
              "🔴 históricamente falla · 🟢 ventaja (rara) · ⚪ sin ventaja\n"]
    for h in top:
        cuerpo.append(f"{_emoji(h['figura'], vd)} {h['nombre']} · {h['ticker']} "
                      f"{h['precio']:.2f} · 💪{h['fuerza']}")
    pie = ("🌐 tristansuarez.github.io/neural-capital-research\n"
           "⚠️ Educativo. No es recomendación de inversión.")
    for trozo in esc.trocear_con_pie("\n".join(cuerpo), pie):
        if args.telegram:
            esc.enviar_telegram(trozo)
        else:
            print(trozo)


if __name__ == "__main__":
    main()
