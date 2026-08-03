# -*- coding: utf-8 -*-
"""
analista_mineras.py — Analista automático LOCAL de noticias + chat sobre el laboratorio.
========================================================================================
Corre en tu máquina (Windows/PowerShell) con Ollama. NO corre en Actions: un chat
público en GitHub Pages exigiría exponer una clave o pagar servidor, así que la
parte inteligente vive en local y solo sus VEREDICTOS viajan al repo, donde son
públicos y auditables.

DOS MODOS:

  python analista_mineras.py
      Analiza: baja los titulares recientes de cada ticker de mineras_universo.txt,
      los clasifica con Ollama usando un prompt FIJO (pre-registrado abajo, no se
      adapta según lo que salga), escribe el log completo en noticias_mineras.csv
      y rellena la columna `notas` de forward_mineras.csv para las elegidas del
      mes en curso (solo si está vacía: las notas ya escritas no se retocan).
      Después: git add + commit + push de los dos CSV y la web las mostrará.

  python analista_mineras.py --chat
      Pregunta: carga resultados.json y los CSV de forward y abre un chat con
      Ollama que responde SOLO desde esos datos ("¿por qué se rechazó FNV?",
      "¿qué noticias tiene BTG?"). Si algo no está en los datos, debe decirlo.

  Opciones: --modelo NOMBRE (si no, usa el primero que liste Ollama)
            --max-noticias N (por ticker, por defecto 5)

PROMPT DE CLASIFICACIÓN (pre-registrado; cambiarlo es un cambio de método y se
declara en un commit propio, no se ajusta en silencio):
  Categorías: dilucion | operacional | permisos | fusiones | resultados |
              financiacion | otro
  Signo: -1 (malo para el accionista), 0 (neutro), +1 (bueno)

Requisitos locales:  pip install yfinance requests
Ollama sirviendo en http://localhost:11434 (arranca con `ollama serve` si no).
Operaciones ficticias, registro público. NO es asesoramiento.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

OLLAMA = "http://localhost:11434"
LOG_NOTICIAS = "noticias_mineras.csv"
LOG_FORWARD = "forward_mineras.csv"
UNIVERSO_TXT = "mineras_universo.txt"
RESULTADOS = "resultados.json"

CATEGORIAS = ["dilucion", "operacional", "permisos", "fusiones",
              "resultados", "financiacion", "otro"]

LOG_VEREDICTOS = "veredictos_mineras.csv"

PROMPT_ANALISTA = (
    "Eres un analista fundamental de mineras de oro, escéptico y directo. Se te dan "
    "los datos VIGENTES de una empresa (balance, caja, EBITDA, FCF, valoración, momento "
    "del precio y noticias del mes). Analiza SOLO con esos datos: si algo no está, no lo "
    "supongas. Juzga: salud del balance, si la valoración compensa, y qué puede salir mal. "
    "Responde SOLO con JSON exacto: {\"puntuacion\": 0-10, \"veredicto\": "
    "\"compraria\"|\"mantendria\"|\"evitaria\", \"tesis\": \"2-3 frases con tu "
    "razonamiento\", \"riesgo\": \"el mayor riesgo en 1 frase\"}"
)

PROMPT_INFORME = (
    "Eres un analista de bolsa escéptico y directo, del laboratorio Neural Capital "
    "Research, cuya marca es la honestidad: publicamos también lo que no funciona. Se te "
    "dan TODOS los datos disponibles de un valor. Escribe un informe en español, en "
    "markdown, con EXACTAMENTE estas secciones: "
    "## Resumen (3-4 frases con lo esencial y el precio) · "
    "## Fundamentales (tabla con las métricas dadas y qué significan) · "
    "## Consenso de analistas (qué dice y cuánto fiarse) · "
    "## Técnicos (RSI, MACD, medias: qué señalan y en qué plazo) · "
    "## Noticias (si las hay) · "
    "## Tesis alcista y bajista (viñetas honestas en ambos lados) · "
    "## Veredicto (tu opinión clara, el mayor riesgo, y una nota 0-10). "
    "REGLAS: usa SOLO los datos dados; si una métrica falta, di 'sin dato', no la "
    "inventes; distingue siempre hechos de tu opinión; nada de precios objetivo propios; "
    "cierra con: 'Opinión de un modelo de lenguaje local con datos del día. No es "
    "recomendación de inversión.'"
)

PROMPT_CLASIFICADOR = (
    "Eres un analista de mineras de oro. Clasifica el titular en UNA categoría: "
    + ", ".join(CATEGORIAS) +
    ". Asigna signo para el accionista: -1 malo, 0 neutro, 1 bueno. "
    "dilucion = ampliaciones, ofertas de acciones, warrants, ATM. "
    "operacional = producción, costes, leyes del mineral, accidentes, huelgas. "
    "permisos = licencias, gobiernos, litigios, jurisdicción. "
    "fusiones = M&A, adquisiciones, ventas de activos. "
    "resultados = cuentas trimestrales, guidance, dividendos. "
    "financiacion = deuda, streams, royalties nuevos, refinanciación. "
    "Responde SOLO con JSON: {\"categoria\": \"...\", \"signo\": -1|0|1}"
)


# ------------------------------------------------------------- ollama ----
def _modelos_disponibles():
    import requests
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _elegir_modelo(pedido):
    disp = _modelos_disponibles()
    if not disp:
        sys.exit("No se puede hablar con Ollama en %s. ¿Está corriendo? (ollama serve)" % OLLAMA)
    if pedido:
        if any(m == pedido or m.startswith(pedido + ":") for m in disp):
            return next(m for m in disp if m == pedido or m.startswith(pedido + ":"))
        sys.exit(f"Modelo '{pedido}' no encontrado. Disponibles: {', '.join(disp)}")
    return disp[0]


def _generar(modelo, prompt, sistema=None, contexto_chat=None):
    import requests
    if contexto_chat is not None:
        msgs = ([{"role": "system", "content": sistema}] if sistema else []) + contexto_chat
        r = requests.post(f"{OLLAMA}/api/chat",
                          json={"model": modelo, "messages": msgs, "stream": False},
                          timeout=300)
        return r.json().get("message", {}).get("content", "")
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": modelo, "prompt": prompt,
                            "system": sistema or "", "stream": False,
                            "options": {"temperature": 0.0}},
                      timeout=300)
    return r.json().get("response", "")


def _clasificar(modelo, titular):
    """Devuelve (categoria, signo) o (None, None) si el modelo no responde en formato."""
    salida = _generar(modelo, f"Titular: {titular}", sistema=PROMPT_CLASIFICADOR)
    m = re.search(r"\{.*?\}", salida, re.DOTALL)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
        cat = str(d.get("categoria", "")).lower().strip()
        cat = cat if cat in CATEGORIAS else None
        signo = int(d.get("signo"))
        signo = signo if signo in (-1, 0, 1) else None
        return cat, signo
    except Exception:
        return None, None


# ------------------------------------------------------------ noticias ----
def _universo():
    if not os.path.exists(UNIVERSO_TXT):
        sys.exit(f"No encuentro {UNIVERSO_TXT}: corre esto desde la carpeta del repo.")
    with open(UNIVERSO_TXT, encoding="utf-8") as fh:
        return [ln.strip().upper() for ln in fh
                if ln.strip() and not ln.strip().startswith("#")]


def _titulares(tk, maximo):
    import yfinance as yf
    try:
        noticias = yf.Ticker(tk).news or []
    except Exception:
        return []
    out = []
    for n in noticias[:maximo]:
        c = n.get("content", n) if isinstance(n, dict) else {}
        titulo = (c.get("title") or n.get("title") or "").strip()
        if titulo:
            out.append(titulo)
    return out


def _ya_registradas():
    vistos = set()
    if os.path.exists(LOG_NOTICIAS):
        with open(LOG_NOTICIAS, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vistos.add((row.get("ticker"), row.get("titular")))
    return vistos


def analizar(modelo, max_noticias):
    vistos = _ya_registradas()
    nuevo = not os.path.exists(LOG_NOTICIAS)
    mes = dt.date.today().strftime("%Y-%m")
    resumen = {}
    n_clasificadas = 0
    with open(LOG_NOTICIAS, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["mes", "fecha", "ticker", "titular", "categoria", "signo",
                        "modelo", "registrado"])
        for tk in _universo():
            for titular in _titulares(tk, max_noticias):
                if (tk, titular) in vistos:
                    continue
                cat, signo = _clasificar(modelo, titular)
                if cat is None:
                    cat, signo = "otro", 0   # el fallo de formato se registra como tal
                w.writerow([mes, dt.date.today().isoformat(), tk, titular, cat, signo,
                            modelo, dt.datetime.now(dt.timezone.utc).isoformat()])
                resumen.setdefault(tk, []).append((cat, signo))
                n_clasificadas += 1
                print(f"  {tk}: [{cat} {signo:+d}] {titular[:70]}")
    _anotar_forward(mes, _resumen_del_mes(mes))
    print(f"\n{n_clasificadas} titulares nuevos clasificados por {modelo}.")
    print("Ahora: git add noticias_mineras.csv forward_mineras.csv ; git commit ; git push")


def _resumen_del_mes(mes):
    """Resumen por ticker con TODO lo clasificado este mes (no solo esta pasada):
    así relanzar el analista puede anotar el forward aunque no haya titulares nuevos."""
    resumen = {}
    if not os.path.exists(LOG_NOTICIAS):
        return resumen
    with open(LOG_NOTICIAS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("mes") != mes:
                continue
            try:
                resumen.setdefault(r["ticker"], []).append((r["categoria"], int(r["signo"])))
            except Exception:
                continue
    return resumen


def _anotar_forward(mes, resumen):
    """Rellena `notas` de las elegidas del mes SI está vacía. No retoca nada más."""
    if not os.path.exists(LOG_FORWARD) or not resumen:
        return
    with open(LOG_FORWARD, encoding="utf-8") as fh:
        lector = csv.DictReader(fh)
        campos = lector.fieldnames
        filas = list(lector)
    cambiadas = 0
    for r in filas:
        if r.get("mes") != mes or r.get("notas"):
            continue
        cats = resumen.get(r.get("ticker"))
        if not cats:
            continue
        cuenta = {}
        for c, s in cats:
            cuenta[c] = cuenta.get(c, 0) + s
        r["notas"] = " · ".join(f"{c}{'+' if v > 0 else '-' if v < 0 else '='}"
                                for c, v in sorted(cuenta.items()))
        cambiadas += 1
    if cambiadas:
        with open(LOG_FORWARD, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print(f"notas añadidas a {cambiadas} elegida(s) del mes en {LOG_FORWARD}")


# ----------------------------------------------------------- veredictos ----
def _contexto_empresa(tk, d, precios6m, noticias_mes):
    partes = [f"Empresa: {tk}", f"Precio actual: {d['precio']} $"]
    neta = d["deuda"] - d["caja"]
    partes.append(f"Deuda neta: {neta/1e6:.0f} M$ (deuda {d['deuda']/1e6:.0f} / caja {d['caja']/1e6:.0f})")
    partes.append(f"EBITDA: {'%.0f M$' % (d['ebitda']/1e6) if d['ebitda'] is not None else 'sin dato'}")
    partes.append(f"FCF: {'%.0f M$' % (d['fcf']/1e6) if d['fcf'] is not None else 'sin dato'}")
    if d.get("ev") and d.get("ebitda"):
        partes.append(f"EV/EBITDA: {d['ev']/d['ebitda']:.1f}")
    if d.get("pb") is not None:
        partes.append(f"Precio/valor contable: {d['pb']}")
    if tk in precios6m:
        v1, v6 = precios6m[tk]
        partes.append(f"Precio: {v1:+.1f}% en 1 mes, {v6:+.1f}% en 6 meses")
    ns = noticias_mes.get(tk, [])
    if ns:
        partes.append("Noticias del mes: " + " | ".join(f"[{c} {s:+d}] {t[:80]}" for t, c, s in ns[:5]))
    else:
        partes.append("Noticias del mes: ninguna registrada")
    return "\n".join(partes)


def _precios_6m(tickers, sintetico=False):
    if sintetico:
        import numpy as np
        rng = np.random.default_rng(11)
        return {tk: (float(rng.normal(0, 5)), float(rng.normal(0, 15))) for tk in tickers}
    import yfinance as yf
    out = {}
    try:
        df = yf.download(tickers, period="7mo", auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
        for tk in tickers:
            try:
                s = df[tk]["Close"].dropna()
                if len(s) > 40:
                    out[tk] = (float(s.iloc[-1] / s.iloc[-21] - 1) * 100,
                               float(s.iloc[-1] / s.iloc[0] - 1) * 100)
            except Exception:
                continue
    except Exception:
        pass
    return out


def _noticias_del_mes(mes):
    out = {}
    if not os.path.exists(LOG_NOTICIAS):
        return out
    with open(LOG_NOTICIAS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("mes") == mes:
                try:
                    out.setdefault(r["ticker"], []).append(
                        (r["titular"], r["categoria"], int(r["signo"])))
                except Exception:
                    continue
    return out


def _veredicto_llm(modelo, contexto):
    salida = _generar(modelo, contexto, sistema=PROMPT_ANALISTA)
    m = re.search(r"\{.*\}", salida, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        p = float(d.get("puntuacion"))
        v = str(d.get("veredicto", "")).lower().strip()
        if not (0 <= p <= 10) or v not in ("compraria", "mantendria", "evitaria"):
            return None
        return {"puntuacion": round(p, 1), "veredicto": v,
                "tesis": str(d.get("tesis", ""))[:400].replace("\n", " "),
                "riesgo": str(d.get("riesgo", ""))[:200].replace("\n", " ")}
    except Exception:
        return None


def veredictos(modelo, sintetico=False):
    """El analista de verdad: tesis, riesgo y puntuación por minera, con datos
    delante. La OPINIÓN se publica como opinión; la PUNTUACIÓN se mide: con el
    tiempo, el propio CSV dirá si las notas altas baten a las bajas."""
    import forward_mineras as fm
    datos, _bench, errores = fm._fundamentales(sintetico)
    if not datos:
        sys.exit("Sin fundamentales (¿red?).")
    mes = dt.date.today().strftime("%Y-%m")
    ya = set()
    if os.path.exists(LOG_VEREDICTOS):
        with open(LOG_VEREDICTOS, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("mes") == mes:
                    ya.add(r.get("ticker"))
    precios6m = _precios_6m(list(datos), sintetico)
    noticias = _noticias_del_mes(mes)
    nuevo = not os.path.exists(LOG_VEREDICTOS)
    n_ok, n_mal = 0, 0
    with open(LOG_VEREDICTOS, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(["mes", "fecha", "ticker", "precio", "puntuacion", "veredicto",
                        "tesis", "riesgo", "modelo", "registrado"])
        for tk, d in datos.items():
            if tk in ya:
                continue
            v = _veredicto_llm(modelo, _contexto_empresa(tk, d, precios6m, noticias))
            ts = dt.datetime.now(dt.timezone.utc).isoformat()
            if v is None:
                w.writerow([mes, dt.date.today().isoformat(), tk, d["precio"], "", "",
                            "sin formato: el modelo no devolvio JSON valido", "", modelo, ts])
                n_mal += 1
                print(f"  {tk}: (sin formato)")
                continue
            w.writerow([mes, dt.date.today().isoformat(), tk, d["precio"],
                        v["puntuacion"], v["veredicto"], v["tesis"], v["riesgo"], modelo, ts])
            n_ok += 1
            print(f"  {tk}: {v['puntuacion']}/10 {v['veredicto']} — {v['tesis'][:70]}")
    print(f"\n{n_ok} veredictos ({n_mal} sin formato) de {modelo}. Tickers sin datos: {len(errores)}.")
    print("Ahora: git add veredictos_mineras.csv ; git commit ; git push")


# -------------------------------------------------------------- informe ----
def _tecnicos(cierre):
    """RSI(14), MACD(12,26,9), medias 50/200 y posición en el rango de 52 semanas."""
    import numpy as np
    c = cierre.dropna()
    if len(c) < 60:
        return {}
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    senal = macd.ewm(span=9, adjust=False).mean()
    out = {"rsi14": round(rsi, 1),
           "macd": round(float(macd.iloc[-1]), 3),
           "macd_senal": round(float(senal.iloc[-1]), 3),
           "macd_hist": round(float(macd.iloc[-1] - senal.iloc[-1]), 3)}
    if len(c) >= 200:
        out["sma50"] = round(float(c.rolling(50).mean().iloc[-1]), 2)
        out["sma200"] = round(float(c.rolling(200).mean().iloc[-1]), 2)
        out["precio_vs_sma200"] = "por encima" if c.iloc[-1] > out["sma200"] else "por debajo"
    ult = c[c.index >= c.index[-1] - pd_timedelta_dias(365)]
    if len(ult) > 50:
        lo, hi = float(ult.min()), float(ult.max())
        out["rango_52s"] = f"{lo:.2f}-{hi:.2f}"
        out["posicion_52s_pct"] = round((float(c.iloc[-1]) - lo) / (hi - lo) * 100, 0) if hi > lo else None
        out["var_12m_pct"] = round((float(c.iloc[-1]) / float(ult.iloc[0]) - 1) * 100, 1)
    return out


def pd_timedelta_dias(n):
    import pandas as pd
    return pd.Timedelta(days=n)


def _datos_informe(tk, sintetico=False):
    if sintetico:
        import numpy as np, pandas as pd
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2024-01-01", periods=420)
        c = pd.Series(20 * np.exp(np.cumsum(rng.normal(2e-4, 0.02, len(idx)))), idx)
        info = {"currentPrice": round(float(c.iloc[-1]), 2), "marketCap": 3.1e9,
                "trailingPE": None, "forwardPE": -17.0, "priceToBook": 4.2,
                "totalCash": 4.5e8, "totalDebt": 2e6, "ebitda": -2e8,
                "freeCashflow": -7.5e8, "revenueGrowth": -0.4, "profitMargins": -8.0,
                "beta": 2.25, "targetMeanPrice": 13.9, "targetLowPrice": 7.0,
                "targetHighPrice": 21.0, "numberOfAnalystOpinions": 18,
                "recommendationKey": "hold"}
        return info, c
    import yfinance as yf
    t = yf.Ticker(tk)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        c = t.history(period="2y", auto_adjust=True)["Close"].dropna()
    except Exception:
        c = None
    if not info and (c is None or not len(c)):
        return None, None
    return info, c


def _num(v, div=1.0, dec=2):
    try:
        return round(float(v) / div, dec)
    except Exception:
        return None


def informe(modelo, tk, sintetico=False):
    tk = tk.upper()
    info, cierre = _datos_informe(tk, sintetico)
    if info is None:
        sys.exit(f"Yahoo no sirve datos de {tk}.")
    partes = [f"VALOR: {tk}",
              f"Precio: {info.get('currentPrice') or info.get('regularMarketPrice') or 'sin dato'} $",
              f"Capitalización: {_num(info.get('marketCap'), 1e9)} mil M$"]
    f = {"PER (trailing)": info.get("trailingPE"), "PER (forward)": info.get("forwardPE"),
         "Precio/valor contable": info.get("priceToBook"),
         "Caja total (M$)": _num(info.get("totalCash"), 1e6, 0),
         "Deuda total (M$)": _num(info.get("totalDebt"), 1e6, 0),
         "EBITDA (M$)": _num(info.get("ebitda"), 1e6, 0),
         "FCF (M$)": _num(info.get("freeCashflow"), 1e6, 0),
         "Crecimiento de ingresos": info.get("revenueGrowth"),
         "Margen neto": info.get("profitMargins"),
         "Beta": info.get("beta")}
    partes.append("FUNDAMENTALES: " + " | ".join(
        f"{k}: {v if v is not None else 'sin dato'}" for k, v in f.items()))
    n_op = info.get("numberOfAnalystOpinions")
    partes.append("CONSENSO DE ANALISTAS: " + (
        f"{n_op} analistas, recomendación '{info.get('recommendationKey','sin dato')}', "
        f"precio objetivo medio {info.get('targetMeanPrice','sin dato')} $ "
        f"(rango {info.get('targetLowPrice','?')}-{info.get('targetHighPrice','?')} $)"
        if n_op else "sin dato"))
    tec = _tecnicos(cierre) if cierre is not None else {}
    partes.append("TÉCNICOS (diario): " + (" | ".join(f"{k}: {v}" for k, v in tec.items())
                                           if tec else "sin dato"))
    mes = dt.date.today().strftime("%Y-%m")
    ns = _noticias_del_mes(mes).get(tk, [])
    partes.append("NOTICIAS DEL MES (clasificadas): " + (
        " | ".join(f"[{c} {s:+d}] {t[:90]}" for t, c, s in ns[:6]) if ns else "ninguna registrada"))
    # contexto del laboratorio si el valor está en nuestro registro
    lab = []
    if os.path.exists(LOG_FORWARD):
        with open(LOG_FORWARD, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("ticker") == tk:
                    lab.append(f"en la selección del forward de {r['mes']} a {r['precio']} $ ({r['metrica_orden']})")
    partes.append("CONTEXTO DEL LABORATORIO: " + ("; ".join(lab[-3:]) if lab else
                  "no está en la selección actual del forward de mineras"))

    contexto = "\n".join(partes)
    print(f"Generando informe de {tk} con {modelo}...\n")
    salida = _generar(modelo, contexto, sistema=PROMPT_INFORME)
    os.makedirs("informes", exist_ok=True)
    ruta = os.path.join("informes", f"{tk}_{dt.date.today().isoformat()}.md")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(f"# Informe {tk} · {dt.date.today().isoformat()} · modelo {modelo}\n\n")
        fh.write(salida.strip() + "\n")
    print(salida.strip())
    print(f"\nGuardado en {ruta}. Si quieres publicarlo: git add informes ; git commit ; git push")


# ---------------------------------------------------------------- chat ----
def _contexto_laboratorio():
    partes = []
    if os.path.exists(RESULTADOS):
        try:
            d = json.load(open(RESULTADOS, encoding="utf-8"))
            partes.append(f"Laboratorio generado: {d.get('generado')}")
            for e in d.get("experimentos", []):
                lin = [f"## {e.get('etiqueta')} ({e.get('tipo','')})"]
                for b in e.get("figuras", []) or []:
                    for p in b.get("puntos", []) or []:
                        lin.append(f"- {b.get('tipo', b.get('nombre'))}: "
                                   f"{p.get('valor')}%/mes p={p.get('p')} "
                                   f"FDR={'sí' if p.get('sig_fdr') else 'no'}")
                    extra = re.sub("<[^>]+>", " ", b.get("extra", ""))
                    if extra.strip():
                        lin.append(f"  {extra.strip()}")
                nota = re.sub("<[^>]+>", " ", (e.get("nota") or "") + " " + (e.get("intro") or ""))
                lin.append(nota[:600])
                partes.append("\n".join(lin))
        except Exception:
            partes.append("(resultados.json ilegible)")
    for csv_f, titulo in ((LOG_FORWARD, "Registro forward de mineras"),
                          ("forward_carteras.csv", "Registro forward de carteras"),
                          (LOG_NOTICIAS, "Noticias clasificadas"),
                          (LOG_VEREDICTOS, "Veredictos del analista")):
        if os.path.exists(csv_f):
            with open(csv_f, encoding="utf-8") as fh:
                filas = fh.read().splitlines()
            partes.append(f"## {titulo} (CSV)\n" + "\n".join(filas[:1] + filas[-120:]))
    return "\n\n".join(partes)


SISTEMA_CHAT = (
    "Eres el analista del laboratorio Neural Capital Research. Respondes SOLO con los "
    "datos del contexto que sigue (resultados de backtests, registros forward y noticias "
    "clasificadas). Si la respuesta no está en los datos, dilo claramente en vez de "
    "inventar. Sé conciso y honesto: este laboratorio publica también lo que no funciona. "
    "Nada de lo que digas es recomendación de inversión.\n\nDATOS:\n"
)


def chat(modelo):
    contexto = _contexto_laboratorio()
    print(f"Chat con {modelo} sobre el laboratorio. Escribe 'salir' para terminar.\n")
    historia = []
    while True:
        try:
            preg = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not preg or preg.lower() in ("salir", "exit", "quit"):
            break
        historia.append({"role": "user", "content": preg})
        resp = _generar(modelo, "", sistema=SISTEMA_CHAT + contexto,
                        contexto_chat=historia[-8:])
        print(f"\nanalista> {resp.strip()}\n")
        historia.append({"role": "assistant", "content": resp})


# ---------------------------------------------------------------- main ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--veredicto", action="store_true")
    ap.add_argument("--informe", metavar="TICKER", default=None)
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--max-noticias", type=int, default=5)
    args = ap.parse_args()
    modelo = _elegir_modelo(args.modelo)
    print(f"Modelo: {modelo}")
    if args.chat:
        chat(modelo)
    elif args.veredicto:
        veredictos(modelo, sintetico=args.sintetico)
    elif args.informe:
        informe(modelo, args.informe, sintetico=args.sintetico)
    else:
        analizar(modelo, args.max_noticias)
