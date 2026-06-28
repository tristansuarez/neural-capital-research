#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
senal_figuras_intradia.py — Avisos de figuras en velas de 1 hora (INFORMATIVO).
==============================================================================
Escanea un universo acotado del S&P 500 en velas de 1 hora, detecta figuras
formadas en la ULTIMA vela y publica en Telegram con el veredicto del backtest
intradia (resultados.json -> figuras_intradia).

SEND-ONLY: no escribe ni commitea nada, para no disparar el laboratorio ni
generar commits cada 30 minutos. Por eso (de momento) puede repetir un aviso si
la misma figura sigue en la ultima vela en la corrida siguiente; el dedup entre
corridas se añadira mas adelante.

Limitaciones honestas: el historico horario gratuito es corto, GitHub Actions no
es tiempo real (se retrasa) y Yahoo limita; esto es contexto, NO una señal.
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
import yfinance as yf

import figuras
import escaner_senales_telegram as esc

UNIVERSO = 120   # nº de valores escaneados (acotado para no saturar Yahoo)
UMBRAL_FUERZA = 35   # solo rupturas decisivas


def _bar_fresca(ts, horas=3):
    """True si la última vela cerró hace <= 'horas'. Descarta festivos y mercado cerrado
    sin necesidad de un calendario: si el mercado está cerrado, la última vela es vieja."""
    try:
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return (pd.Timestamp.now(tz="UTC") - t) <= pd.Timedelta(hours=horas)
    except Exception:
        return True


def _veredictos():
    try:
        with open("resultados.json", encoding="utf-8") as fh:
            d = json.load(fh)
        fig = next(e for e in d["experimentos"] if e.get("id") == "figuras_intradia")
    except Exception:
        return {}
    out = {}
    for f in fig.get("figuras", []):
        p = next((p for p in f["puntos"] if p["etiqueta"] == "2 sem"), None) \
            or (f["puntos"][-1] if f["puntos"] else None)
        if p:
            out[f["tipo"]] = (p["valor"], bool(p.get("sig_fdr", False)))
    return out


def _frase(tipo, vd):
    if tipo not in vd:
        return "histórico intradía: aún sin backtest"
    val, sig = vd[tipo]
    if sig and val < 0:
        return f"histórico 2sem: {val:+.1f}% — tiende a FALLAR (el precio suele ir al contrario)"
    if sig and val > 0:
        return f"histórico 2sem: {val:+.1f}% — ventaja a favor (rara; con cautela)"
    return f"histórico 2sem: {val:+.1f}% — sin ventaja fiable (abraza el cero)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    args = ap.parse_args()

    vd = _veredictos()
    tickers = args.tickers or esc.obtener_sp500()[:UNIVERSO]
    hall = []
    for j in range(0, len(tickers), 40):
        chunk = tickers[j:j + 40]
        try:
            df = yf.download(chunk, period="60d", interval="1h", group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for tk in chunk:
            try:
                sub = df[tk].dropna()
                if len(sub) < 120:
                    continue
                if not _bar_fresca(sub.index[-1]):
                    continue   # festivo / fuera de mercado: la última vela no es de ahora
                H = np.asarray(sub["High"].values, float)
                L = np.asarray(sub["Low"].values, float)
                C = np.asarray(sub["Close"].values, float)
                hora = sub.index[-1].strftime("%Y-%m-%d %H:%M")
                ev = figuras.detectar(H, L, C)
            except Exception:
                continue
            n = len(C)
            for (i, tipo, _d) in ev:
                if i >= n - 1:
                    fz = figuras.fuerza_figura(H, L, C, i)
                    if fz < UMBRAL_FUERZA:
                        continue
                    hall.append({"ticker": tk, "hora": hora, "figura": tipo,
                                 "nombre": figuras.FIGURAS[tipo][0], "precio": float(C[-1]),
                                 "fuerza": int(fz)})

    print(f"Figuras intradía en la última vela: {len(hall)}")
    if not hall:
        return

    hora = hall[0]["hora"]

    def _emoji(tipo):
        vs = vd.get(tipo)
        if vs:
            val, sig = vs
            if sig and val < 0:
                return "🔴"
            if sig and val > 0:
                return "🟢"
        return "⚪"

    top = sorted(hall, key=lambda h: (0 if _emoji(h["figura"]) != "⚪" else 1,
                                      -h.get("fuerza", 0)))
    cuerpo = [f"⏱️ FIGURAS INTRADÍA (1h) · {hora}",
              f"Todas las decisivas de la última vela ({len(top)}). Informativo, no señal: "
              f"las rupturas tienden a revertir.",
              "🔴 históricamente falla · 🟢 ventaja (rara) · ⚪ sin ventaja\n"]
    for h in top:
        cuerpo.append(f"{_emoji(h['figura'])} {h['nombre']} · {h['ticker']} "
                      f"{h['precio']:.2f} · 💪{h.get('fuerza', 0)}")
    pie = ("🌐 tristansuarez.github.io/neural-capital-research\n"
           "⚠️ Educativo. No es recomendación de inversión.")
    for trozo in esc.trocear_con_pie("\n".join(cuerpo), pie):
        if args.telegram:
            esc.enviar_telegram(trozo)
        else:
            print(trozo)


if __name__ == "__main__":
    main()
