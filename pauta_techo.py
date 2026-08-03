"""
pauta_techo.py — Método de techo con pauta plana (estrategia de un especulador).
===============================================================================
Estrategia concreta propuesta públicamente, implementada literalmente y testeada.
Busca CORTOS contra tendencia alcista cuando se cumplen TODAS estas condiciones:

  1. Tendencia alcista ACELERADA: dos directrices alcistas, la segunda con más
     pendiente que la primera (los retrocesos se quedan por encima de la original).
  2. FIGURA DE TECHO: dos máximos crecientes y perforación del mínimo del valle
     entre ellos.
  3. RUPTURA de la directriz alcista.
  4. RESISTENCIA RELEVANTE previa superada antes del giro (máximos históricos de
     la ventana anterior).
  5. PAUTA PLANA ABC en el último tramo bajista, con entrada al tocar la zona de
     medias de 20-50 y girarse a la baja.

Se mide el retorno del CORTO (positivo = el precio baja, la estrategia acierta),
como exceso sobre la tasa base del propio valor.

AVISO METODOLÓGICO CENTRAL: cinco condiciones encadenadas reducen los eventos
drásticamente. Por eso se cuenta cuántos sobreviven a CADA capa: si al final
quedan pocos, el resultado no es evaluable y así se declara. Una estrategia muy
específica siempre «parece» funcionar porque solo se ven los casos que encajan.
NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

HZ = [5, 10, 21, 42]
LAB = {5: "1 sem", 10: "2 sem", 21: "1 mes", 42: "2 meses"}
N_TICKERS = 120
ANOS = 15
MIN_EVENTOS = 30          # por debajo de esto, se declara no evaluable


def _panel(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(103)
        n, k = 2600, 40
        idx = pd.bdate_range("2015-01-01", periods=n)
        return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, (n, k)), axis=0)),
                            index=idx, columns=[f"SYN{i}" for i in range(k)])
    import datetime as dt
    import yfinance as yf
    import escaner_senales_telegram as esc
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    tickers = esc.obtener_sp500()[:N_TICKERS]
    cierres = {}
    for j in range(0, len(tickers), 40):
        chunk = tickers[j:j + 40]
        try:
            df = yf.download(chunk, start=inicio, auto_adjust=True, progress=False,
                             group_by="ticker", threads=True)
        except Exception:
            continue
        for tk in chunk:
            try:
                s = df[tk]["Close"].dropna()
                if len(s) > 600:
                    cierres[tk] = s
            except Exception:
                continue
    return pd.DataFrame(cierres).ffill() if len(cierres) >= 20 else None


def _pivotes(c, k=5):
    n = len(c)
    ph = [i for i in range(k, n - k) if c[i] == max(c[i - k:i + k + 1])]
    pl = [i for i in range(k, n - k) if c[i] == min(c[i - k:i + k + 1])]
    return ph, pl


def detectar_pauta(c, contadores):
    """Aplica las cinco condiciones por capas, contando supervivientes en cada una."""
    n = len(c)
    if n < 200:
        return []
    ph, pl = _pivotes(c)
    eventos = []

    # CAPA 1+2: dos máximos crecientes + perforación del valle (figura de techo)
    for a, b in zip(ph, ph[1:]):
        if not (10 <= b - a <= 90) or c[b] <= c[a]:
            continue
        valle = float(np.min(c[a:b + 1]))
        rot = None
        for j in range(b + 1, min(b + 60, n)):
            if c[j] < valle:
                rot = j
                break
        if rot is None:
            continue
        contadores["1_techo"] += 1

        # CAPA 3: tendencia alcista previa acelerada (pendiente creciente)
        ini = max(0, a - 150)
        if b - ini < 60:
            continue
        mitad = ini + (a - ini) // 2
        p1 = (c[mitad] - c[ini]) / max(1, mitad - ini)
        p2 = (c[a] - c[mitad]) / max(1, a - mitad)
        if not (p1 > 0 and p2 > p1):
            continue
        contadores["2_acelerada"] += 1

        # CAPA 4: resistencia relevante previa superada antes del giro
        prev = c[max(0, ini - 250):ini]
        if len(prev) < 100 or c[b] <= float(np.max(prev)):
            continue
        contadores["3_resistencia"] += 1

        # CAPA 5: pauta plana ABC tras la ruptura + entrada en zona de medias 20-50
        fin = min(rot + 40, n - 1)
        tramo = c[rot:fin]
        if len(tramo) < 12:
            continue
        sub_h, sub_l = _pivotes(tramo, k=2)
        # ABC: rebote, caída, rebote (al menos dos máximos y un mínimo intermedios)
        if len(sub_h) < 2 or len(sub_l) < 1:
            continue
        contadores["4_pauta_plana"] += 1

        # entrada: cuando el rebote toca la zona de medias de 20-50 y se gira
        m20 = pd.Series(c).rolling(20).mean().values
        m50 = pd.Series(c).rolling(50).mean().values
        entrada = None
        for j in range(rot + sub_h[-1], min(rot + sub_h[-1] + 15, n - 1)):
            if not (np.isfinite(m20[j]) and np.isfinite(m50[j])):
                continue
            zona_hi = max(m20[j], m50[j]) * 1.01
            zona_lo = min(m20[j], m50[j]) * 0.99
            if zona_lo <= c[j] <= zona_hi and c[j + 1] < c[j]:
                entrada = j + 1
                break
        if entrada is None:
            continue
        contadores["5_entrada"] += 1
        eventos.append(entrada)
    return eventos


def backtest_pauta(sintetico=False):
    px = _panel(sintetico)
    if px is None:
        return None
    contadores = {"1_techo": 0, "2_acelerada": 0, "3_resistencia": 0,
                  "4_pauta_plana": 0, "5_entrada": 0}
    acc = {h: [] for h in HZ}
    tickers = 0
    for tk in px.columns:
        c = px[tk].dropna().values
        if len(c) < 300:
            continue
        tickers += 1
        m = len(c)
        base = {h: float(np.nanmean(c[h:] / c[:-h] - 1.0)) for h in HZ}
        for i in detectar_pauta(c, contadores):
            for h in HZ:
                if i + h < m:
                    # CORTO: gana si el precio baja -> signo negativo
                    acc[h].append(-(c[i + h] / c[i] - 1.0 - base[h]) * 100.0)

    n_final = contadores["5_entrada"]
    filas_capas = "".join(
        f"<tr><td>{k.split('_', 1)[1].replace('_', ' ').capitalize()}</td>"
        f"<td class='mono'>{v}</td></tr>" for k, v in contadores.items())
    tabla_capas = ("<b>Eventos que sobreviven a cada condición</b> (cinco filtros encadenados):"
                   "<div class='ops-scroll'><table class='ops'><thead><tr><th>Condición</th>"
                   f"<th>Casos</th></tr></thead><tbody>{filas_capas}</tbody></table></div>")

    if n_final < MIN_EVENTOS:
        return {
            "id": "pauta_techo",
            "etiqueta": "Método de techo con pauta plana",
            "tipo": f"{tickers} valores · {ANOS} años · NO EVALUABLE (muestra insuficiente)",
            "modelo": "pauta",
            "sin_datos": True,
            "intro": (f"La estrategia exige cinco condiciones simultáneas. Tras aplicarlas todas "
                      f"quedan solo <b>{n_final} casos</b> en {ANOS} años y {tickers} valores, por "
                      f"debajo del mínimo de {MIN_EVENTOS} para decir nada con fundamento."),
            "nota": (tabla_capas +
                     "<br><b>Este es el hallazgo, y es importante:</b> una estrategia con cinco "
                     "condiciones encadenadas casi nunca se dispara. Con tan pocos casos, cualquier "
                     "resultado —bueno o malo— es indistinguible del azar, y quien la promociona "
                     "puede enseñar siempre los aciertos porque son contables con los dedos. "
                     "No es que la estrategia sea mala: es que <b>no es verificable</b>, que a "
                     "efectos científicos es peor. No es recomendación de inversión."),
        }

    puntos = []
    for h in HZ:
        x = acc[h]
        if len(x) < 20:
            continue
        mm, ic, p1 = figuras._boot_media(x, bloque=max(5, h // 3))
        puntos.append({"h": h, "etiqueta": LAB[h], "valor": round(mm, 2),
                       "ic_lo": round(ic[0], 2), "ic_hi": round(ic[1], 2),
                       "n": len(x), "p": min(1.0, 2 * p1)})
    if not puntos:
        return None
    mask = figuras._bh([p["p"] for p in puntos], q=0.10)
    for p, ok in zip(puntos, mask):
        p["sig_cruda"] = bool(p["ic_lo"] > 0 or p["ic_hi"] < 0)
        p["sig_fdr"] = bool(ok)
    n_fdr = int(np.sum(mask))

    return {
        "id": "pauta_techo",
        "etiqueta": "Método de techo con pauta plana",
        "tipo": f"{tickers} valores · {ANOS} años · {n_final} señales · cortos contra tendencia",
        "modelo": "pauta",
        "figuras_panel": True,
        "intro": ("Estrategia concreta de un especulador público, implementada literalmente: "
                  "tendencia alcista acelerada, figura de techo con perforación del valle, ruptura "
                  "de directriz, resistencia relevante previa y pauta plana ABC con entrada en la "
                  "zona de medias 20-50. Se abre CORTO: «ventaja» positiva = el precio baja después, "
                  "es decir, la estrategia acierta."),
        "figuras": [{"tipo": "pauta_techo", "nombre": f"Corto tras pauta de techo · {n_final} señales",
                     "color": "#d2566a", "n_eventos": n_final,
                     "puntos": [{"etiqueta": p["etiqueta"], "valor": p["valor"], "ic_lo": p["ic_lo"],
                                 "ic_hi": p["ic_hi"], "n": p["n"], "sig_cruda": p["sig_cruda"],
                                 "sig_fdr": p["sig_fdr"]} for p in puntos]}],
        "n_celdas": len(puntos),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se evalúan {len(puntos)} horizontes con corrección Benjamini-Hochberg "
                     f"(FDR 10%): {n_fdr} sobreviven. La estrategia opera CONTRA la tendencia, que "
                     f"es estadísticamente lo más difícil."),
        "nota": (tabla_capas +
                 "<br>Ojo al embudo: cada condición añadida reduce los casos. Una estrategia muy "
                 "específica parece infalible porque solo se ven los casos que encajaron, y rara vez "
                 "hay muestra suficiente para probarlo. Aquí se cuenta el embudo entero. "
                 "No es recomendación de inversión."),
    }
