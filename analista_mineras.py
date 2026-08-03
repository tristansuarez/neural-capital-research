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
    _anotar_forward(mes, resumen)
    print(f"\n{n_clasificadas} titulares nuevos clasificados por {modelo}.")
    print("Ahora: git add noticias_mineras.csv forward_mineras.csv ; git commit ; git push")


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
                          (LOG_NOTICIAS, "Noticias clasificadas")):
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
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--max-noticias", type=int, default=5)
    args = ap.parse_args()
    modelo = _elegir_modelo(args.modelo)
    print(f"Modelo: {modelo}")
    if args.chat:
        chat(modelo)
    else:
        analizar(modelo, args.max_noticias)
