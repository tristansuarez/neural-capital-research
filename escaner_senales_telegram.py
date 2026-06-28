#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
escaner_senales_telegram.py
===========================
Escanea el S&P 500 buscando senales KONCORDE V3 (cruce de Blai5 + regimen de
Fosback) en la ULTIMA sesion disponible, registra cada senal en un CSV (para poder
medir a futuro si funciona, sin sesgo de supervivencia) y la PUBLICA en un canal
de Telegram (gratis).

FLUJO
-----
1. Python (determinista) calcula KONCORDE y detecta que valores dan senal HOY.
2. Cada senal se guarda en senales_log.csv (ticker, fecha, precio, azul, verde).
3. El mensaje se ensambla en plantilla fija (frase + emojis + descargo) y:
   - se guarda en tweets_hoy.txt, y
   - si se pasa --telegram, se publica en el canal via la API de bots (gratis).

CONFIG (variables de entorno, NO escribir en el codigo)
-------------------------------------------------------
  TELEGRAM_TOKEN   token del bot de @BotFather   (export TELEGRAM_TOKEN='...')
  TELEGRAM_CHANNEL canal destino (def. @koncorde_signals)

USO
---
  python escaner_senales_telegram.py                 # solo detecta y redacta
  python escaner_senales_telegram.py --telegram      # ademas publica en el canal
  python escaner_senales_telegram.py --test-telegram # envia un mensaje de prueba y sale
  python escaner_senales_telegram.py --sin-modelo    # sin OpenClaw (plantilla fija, rapido)

HONESTIDAD
----------
La medicion previa mostro que esta senal NO bate al azar. Esto publica senales que,
segun tus propios datos, no anticipan subidas. El CSV existe para que lo COMPRUEBES a
futuro de forma limpia. Si lo ve mas gente, recuerda que el descargo no exime de que
estas emitiendo senales que sabes que no funcionan. Uso responsable.

NO es asesoramiento financiero.
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

# Silenciar los avisos de yfinance por tickers deslistados (ANSS, WBA, etc.)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

try:
    import yfinance as yf
except ImportError:
    sys.exit("Falta yfinance. Instala con: pip install yfinance pandas numpy")

LOG_CSV = "senales_log.csv"
TWEETS_OUT = "tweets_hoy.txt"
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHANNEL = os.environ.get("TELEGRAM_CHANNEL", "@koncorde_signals")

# --- Jerarquia por POTENCIA (intensidad del indicador, NO rentabilidad esperada) ---
# La potencia = area azul (manos fuertes) + area verde (manos debiles) en la senal.
# Mide cuan intensa es la lectura del indicador, no cuanto subira el precio.
# Umbrales CALIBRADOS con una muestra real (potencias ~6 a ~43). Ajustables.
POTENCIA_ALTA = 28
POTENCIA_MEDIA = 17

# Jerarquia visual por tamano/poder: ballena > toro > pez.
EMOJI_NIVEL = {3: "🐋", 2: "🐂", 1: "🐟"}

FRASES_NIVEL = {
    3: [
        "Entrada MUY fuerte de manos fuertes en {ticker}",
        "Acumulación institucional intensa en {ticker}",
        "Lectura KONCORDE de máxima intensidad en {ticker}",
    ],
    2: [
        "Entrada de manos fuertes en {ticker}",
        "El dinero inteligente se posiciona en {ticker}",
        "Acumulación clara en {ticker}",
    ],
    1: [
        "Señal incipiente de manos fuertes en {ticker}",
        "Primeros indicios de acumulación en {ticker}",
        "Activación leve de KONCORDE en {ticker}",
    ],
}


def nivel_potencia(potencia):
    """Devuelve el nivel (3 alta, 2 media, 1 baja) segun la potencia del indicador."""
    if potencia >= POTENCIA_ALTA:
        return 3
    if potencia >= POTENCIA_MEDIA:
        return 2
    return 1

# NASDAQ-100 (lista de partida; los componentes cambian con el tiempo).
NASDAQ100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "PEP","NFLX","ADBE","AMD","CSCO","TMUS","INTC","CMCSA","QCOM","INTU",
    "AMGN","TXN","HON","AMAT","ISRG","BKNG","SBUX","MDLZ","ADI","GILD",
    "ADP","VRTX","REGN","LRCX","MU","PANW","SNPS","CDNS","MELI","PYPL",
    "ASML","KLAC","CRWD","MAR","ABNB","ORLY","CSX","CTAS","NXPI","FTNT",
    "WDAY","CHTR","MNST","ADSK","PCAR","PAYX","ROP","CPRT","ODFL","KDP",
    "MRVL","DXCM","AEP","FAST","ROST","BKR","KHC","EA","CEG","VRSK",
    "EXC","CTSH","XEL","DDOG","TTD","GEHC","IDXX","CCEP","ZS","ANSS",
    "ON","TEAM","DLTR","WBD","BIIB","ILMN","MDB","WBA","ARM","SMCI",
    "LULU","PDD","FANG","TTWO","GFS","SIRI","MRNA","LCID","ENPH","ALGN",
]

# Lista de respaldo del S&P 500 (grandes valores por sector). Se usa SOLO si la
# descarga de la lista actualizada desde Wikipedia falla, para que el bot nunca
# se quede sin universo que escanear.
SP500_FALLBACK = NASDAQ100 + [
    "JPM","V","JNJ","WMT","PG","XOM","HD","KO","CVX","MRK","BAC","DIS","MCD",
    "ABBV","LLY","UNH","MA","PFE","TMO","ABT","CRM","ACN","WFC","DHR","LIN",
    "NKE","TXN","NEE","PM","RTX","UNP","LOW","SPGI","GS","CAT","IBM","BA",
    "GE","MMM","AXP","BLK","ELV","DE","LMT","SYK","C","MDT","CB","MO","CI",
    "SO","DUK","BMY","UPS","T","VZ","CVS","TGT","CL","GD","NOC","USB","PNC",
    "MS","SCHW","BX","TFC","COF","AIG","MET","PRU","TRV","ALL","F","GM",
    "EOG","SLB","PSX","MPC","VLO","OXY","WMB","KMI","D","AEP","SRE","PEG",
    "ED","EIX","DOW","DD","NEM","FCX","APD","SHW","ECL","EMR","ETN","ITW",
    "PH","ROK","CMI","FDX","NSC","WM","RSG","MMC","AON","ICE","CME","PGR",
    "AFL","HUM","CNC","ZTS","BDX","BSX","EW","HCA","DVA","A","IQV","RMD",
]


def obtener_sp500():
    """Obtiene la lista actualizada del S&P 500 desde Wikipedia.
    Si falla (p. ej. sin red o Wikipedia caida), devuelve la lista de respaldo."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        import io
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        html = requests.get(url, headers=headers, timeout=30).text
        tablas = pd.read_html(io.StringIO(html))
        # yfinance usa "-" en vez de "." para acciones de clase (BRK.B -> BRK-B).
        tickers = (tablas[0]["Symbol"].astype(str)
                   .str.replace(".", "-", regex=False).tolist())
        if len(tickers) > 400:
            print(f" Lista S&P 500 obtenida de Wikipedia: {len(tickers)} valores.")
            return tickers
    except Exception as e:
        print(f" No se pudo obtener la lista de Wikipedia ({e}). Usando respaldo.")
    print(f" Usando lista de respaldo: {len(SP500_FALLBACK)} valores.")
    return SP500_FALLBACK


# --------------------------- Indicadores --------------------------- #
def rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0.0); l = -d.clip(upper=0.0)
    ag = g.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = l.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3.0; flow = tp * v
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def stoch_k(h, l, c, n=21):
    ll = l.rolling(n).min(); hh = h.rolling(n).max()
    return 100 * (c - ll) / (hh - ll).replace(0, np.nan)


def boll_pct(c, n=25, k=2.0):
    mid = c.rolling(n).mean(); sd = c.rolling(n).std()
    return (c - (mid - k * sd)) / (2 * k * sd).replace(0, np.nan) * 100


def nvi_pvi(c, v):
    ret = c / c.shift(1)
    pvi = 1000.0 * ret.where(v > v.shift(1), 1.0).fillna(1.0).cumprod()
    nvi = 1000.0 * ret.where(v < v.shift(1), 1.0).fillna(1.0).cumprod()
    return nvi, pvi


def koncorde(df):
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    marron = (rsi(c) + mfi(h, l, c, v) + stoch_k(h, l, c) + boll_pct(c)) / 4.0 - 50.0
    media = marron.ewm(span=15, adjust=False).mean()
    nvi, pvi = nvi_pvi(c, v)
    nvim, pvim = nvi.ewm(span=9, adjust=False).mean(), pvi.ewm(span=9, adjust=False).mean()
    w = 90
    azul = (nvi - nvim) * 100 / (nvim.rolling(w).max() - nvim.rolling(w).min()).replace(0, np.nan)
    verde = (pvi - pvim) * 100 / (pvim.rolling(w).max() - pvim.rolling(w).min()).replace(0, np.nan)
    out = df.copy()
    out["azul"], out["verde"], out["marron"], out["media"] = azul, verde, marron, media
    return out


# --------------------------- Régimen Fosback --------------------------- #
def regimen_alcista(inicio):
    idx = yf.download("^GSPC", start=inicio, auto_adjust=True, progress=False)
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)
    if idx.empty:
        return None
    nvi, _ = nvi_pvi(idx["Close"], idx["Volume"])
    return (nvi > nvi.rolling(252).mean())


# --------------------------- OpenClaw redacta --------------------------- #
def construir_tweet(ticker, precio, frase):
    """Tweet individual (para tweets_hoy.txt). 'frase' ya viene con su emoji de nivel."""
    if not frase:
        frase = f"🔥 Señal KONCORDE en {ticker}"
    return (
        f"🔵📈 SEÑAL KONCORDE · ${ticker}\n"
        f"{frase}\n"
        f"💵 {precio:.2f} · {datetime.now():%d/%m/%Y}\n"
        f"⚠️ No es recomendación de inversión.\n"
        f"#trading #bolsa ${ticker}"
    )


def mensaje_agrupado(lineas):
    """Construye un unico mensaje: cabecera con leyenda, una linea por senal, un descargo."""
    cabecera = (f"📊 SEÑALES KONCORDE · {datetime.now():%d/%m/%Y}\n"
                f"{len(lineas)} valores · ordenados por potencia del indicador\n"
                f"🐋 alta · 🐂 media · 🐟 baja")
    cuerpo = "\n\n".join(lineas)
    pie = ("ℹ️ La potencia mide la intensidad del indicador, NO la rentabilidad esperada.\n"
           "⚠️ Contenido educativo. No es recomendación de inversión. #trading #bolsa")
    return f"{cabecera}\n\n{cuerpo}\n\n{pie}"


def trocear(texto, limite=4000):
    """Parte un mensaje en trozos por parrafos si supera el limite de Telegram (4096)."""
    if len(texto) <= limite:
        return [texto]
    trozos, actual = [], ""
    for parrafo in texto.split("\n\n"):
        if actual and len(actual) + len(parrafo) + 2 > limite:
            trozos.append(actual)
            actual = parrafo
        else:
            actual = f"{actual}\n\n{parrafo}" if actual else parrafo
    if actual:
        trozos.append(actual)
    return trozos


def trocear_con_pie(cuerpo, pie="", limite=3800):
    """Parte 'cuerpo' por lineas en trozos <= limite y pega 'pie' SIEMPRE al ultimo
    trozo (asi el enlace nunca se queda en un mensaje suelto, mida lo que mida)."""
    trozos, actual, n = [], [], 0
    for ln in cuerpo.split("\n"):
        if actual and n + len(ln) + 1 > limite:
            trozos.append("\n".join(actual))
            actual, n = [], 0
        actual.append(ln)
        n += len(ln) + 1
    if actual:
        trozos.append("\n".join(actual))
    if not trozos:
        trozos = [""]
    if pie:
        trozos[-1] = trozos[-1].rstrip() + "\n\n" + pie
    return trozos


# --------------------------- Telegram publica --------------------------- #
def enviar_telegram(texto):
    """Publica 'texto' en el canal de Telegram. Devuelve (ok, detalle)."""
    if not TG_TOKEN:
        return False, "Falta TELEGRAM_TOKEN (export TELEGRAM_TOKEN='...')."
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    datos = urllib.parse.urlencode({
        "chat_id": TG_CHANNEL,
        "text": texto,
        "disable_web_page_preview": "true",
    }).encode()
    try:
        with urllib.request.urlopen(url, data=datos, timeout=30) as resp:
            r = json.loads(resp.read().decode())
            return (r.get("ok", False), "enviado" if r.get("ok") else str(r))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            return False, f"{e.code}: {err.get('description', '')}"
        except Exception:
            return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


# --------------------------- Registro CSV (forward-test) --------------------------- #
def registrar(senales):
    nuevo = not os.path.exists(LOG_CSV)
    ya = set()
    if not nuevo:
        with open(LOG_CSV, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    ya.add((row[0], row[1]))
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if nuevo:
            wr.writerow(["ticker", "fecha", "precio", "azul", "verde", "registrado"])
        n = 0
        for s in senales:
            clave = (s["ticker"], s["fecha"])
            if clave in ya:
                continue
            wr.writerow([s["ticker"], s["fecha"], f"{s['precio']:.4f}",
                         f"{s['azul']:.2f}", f"{s['verde']:.2f}",
                         datetime.now().strftime("%Y-%m-%d %H:%M")])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default="2022-01-01", help="Inicio de la ventana de calculo.")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--sin-modelo", action="store_true",
                    help="No llamar a OpenClaw; usa plantilla fija (mas rapido).")
    ap.add_argument("--telegram", action="store_true",
                    help="Publicar cada senal en el canal de Telegram.")
    ap.add_argument("--test-telegram", action="store_true",
                    help="Enviar un mensaje de prueba al canal y salir.")
    args = ap.parse_args()
    if args.test_telegram:
        ok, det = enviar_telegram("✅ Bot KONCORDE conectado. Canal listo para señales.")
        print(f" Test Telegram -> {'OK, mira tu canal' if ok else 'FALLO: ' + det}")
        return

    tickers = args.tickers if args.tickers else obtener_sp500()

    print("=" * 72)
    print(" ESCANER KONCORDE V3 · S&P 500 · senales de la ultima sesion")
    print("=" * 72)
    print(" Calculando regimen de mercado (NVI del S&P 500)...", flush=True)
    regime = regimen_alcista(args.inicio)
    if regime is None:
        sys.exit(" No se pudo descargar ^GSPC.")
    regime_hoy = bool(regime.dropna().iloc[-1]) if len(regime.dropna()) else False
    print(f" Regimen de mercado: {'ALCISTA ✅' if regime_hoy else 'NO alcista ⛔ (V3 no dispara)'}")
    print("-" * 72)

    senales = []
    for i, tk in enumerate(tickers, 1):
        try:
            df = yf.download(tk, start=args.inicio, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 300:
                continue
            df = koncorde(df)
            reg = regime.reindex(df.index).fillna(False)
            # V3 en la ULTIMA barra: cruce marron sobre su media + azul>0 + verde>0 + regimen.
            marron, media, azul, verde = df["marron"], df["media"], df["azul"], df["verde"]
            cruce = (marron.iloc[-1] > media.iloc[-1]) and (marron.iloc[-2] <= media.iloc[-2])
            cond = cruce and (azul.iloc[-1] > 0) and (verde.iloc[-1] > 0) and bool(reg.iloc[-1])
            if cond:
                senales.append({
                    "ticker": tk,
                    "fecha": df.index[-1].strftime("%Y-%m-%d"),
                    "precio": float(df["Close"].iloc[-1]),
                    "azul": float(azul.iloc[-1]),
                    "verde": float(verde.iloc[-1]),
                    "potencia": float(azul.iloc[-1]) + float(verde.iloc[-1]),
                })
            time.sleep(0.15)
        except Exception as e:
            print(f"  {tk:6s} ERROR: {e}")

    if not senales:
        print(" Hoy NO hay senales V3 en el S&P 500.")
        print(" (Es lo normal: el cruce + regimen es exigente y dispara pocas veces.)")
        print("=" * 72)
        return

    # Ordenar de mayor a menor POTENCIA (intensidad del indicador, NO rentabilidad).
    senales.sort(key=lambda s: s["potencia"], reverse=True)

    n_reg = registrar(senales)
    print(f" Señales detectadas hoy: {len(senales)}  (nuevas en el log: {n_reg})")
    print("-" * 72)

    drafts = []
    lineas = []
    for s in senales:
        nivel = nivel_potencia(s["potencia"])
        emoji = EMOJI_NIVEL[nivel]
        base = None  # frases deterministas de plantilla (FRASES_NIVEL)
        if not base:
            base = random.choice(FRASES_NIVEL[nivel]).format(ticker=s["ticker"])
        frase = f"{emoji} {base}"
        drafts.append(construir_tweet(s["ticker"], s["precio"], frase))
        lineas.append(f"{frase} — 💵 {s['precio']:.2f}")
        print(f" {s['ticker']:6s} pot={s['potencia']:6.1f} niv={nivel}  {base}")

    # Un unico mensaje agrupado, ordenado por potencia, para Telegram.
    if args.telegram:
        mensaje = mensaje_agrupado(lineas)
        for trozo in trocear(mensaje):
            ok, det = enviar_telegram(trozo)
            print(f"   -> Telegram: {'publicado ✅' if ok else 'FALLO: ' + det}")

    with open(TWEETS_OUT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(drafts))
    print(f" {len(drafts)} señal(es) en {TWEETS_OUT}")
    print(f" Señales registradas para forward-test en {LOG_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
