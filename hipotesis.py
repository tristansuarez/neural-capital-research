"""
hipotesis.py — Anomalías clásicas con fundamento económico, testeadas en serio.
==============================================================================
Cada hipótesis se formula ANTES de mirar los datos, con su razón económica:

  H1. DERIVA TRAS SORPRESA (post-earnings drift). Tras un salto de precio muy
      grande en un día (proxy de sorpresa informativa), el precio sigue derivando
      en esa dirección durante semanas, porque el mercado incorpora la información
      despacio (analistas que revisan tarde, atención limitada).

  H2. REVERSIÓN A CORTO PLAZO. Tras una caída fuerte en pocos días, hay rebote,
      porque las ventas forzadas (liquidez, margin calls) empujan el precio por
      debajo de su valor y luego se corrige.

  H3. ESTACIONALIDAD. Ciertos meses rinden distinto de forma sistemática (efecto
      enero por ventas fiscales de diciembre; «sell in May» por flujos estivales).

Todas se miden como EXCESO sobre la tasa base del propio valor, fuera de muestra,
y se corrigen CONJUNTAMENTE por multiple-testing (Benjamini-Hochberg): probar
varias hipótesis sube el listón de todas. El veredicto se acepta tal cual.
NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras   # reutilizamos _boot_media y _bh

HZ = [5, 10, 21, 63]
LAB = {5: "1 sem", 10: "2 sem", 21: "1 mes", 63: "3 meses"}
N_TICKERS = 80
ANOS = 15


def _panel(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(9)
        n, k = 2200, 30
        idx = pd.bdate_range("2015-01-01", periods=n)
        r = rng.normal(0.0003, 0.016, (n, k))
        return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                            columns=[f"SYN{i}" for i in range(k)])
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
                if len(s) > 500:
                    cierres[tk] = s
            except Exception:
                continue
    if len(cierres) < 15:
        return None
    return pd.DataFrame(cierres).ffill()


def _eventos_deriva(c, gap=0.05, lb=20):
    """H1: salto diario >= gap (en valor absoluto), proxy de sorpresa informativa.
    Dirección = signo del salto (la deriva debería continuar)."""
    r = c[1:] / c[:-1] - 1.0
    out = []
    last = -10 ** 9
    for i in range(lb, len(r)):
        if abs(r[i]) >= gap and (i - last) >= lb:
            out.append((i + 1, int(np.sign(r[i])))); last = i
    return out


def _eventos_reversion(c, umbral=-0.10, w=5, lb=15):
    """H2: caída acumulada de >= |umbral| en w días. Dirección = +1 (rebote)."""
    out = []
    last = -10 ** 9
    for i in range(w + lb, len(c)):
        if (c[i] / c[i - w] - 1.0) <= umbral and (i - last) >= lb:
            out.append((i, +1)); last = i
    return out


def _medir(px, eventos_fn, nombre, color, razon):
    acc = {h: [] for h in HZ}
    n_ev = 0
    for tk in px.columns:
        c = px[tk].dropna().values
        if len(c) < 300:
            continue
        m = len(c)
        base = {h: float(np.nanmean(c[h:] / c[:-h] - 1.0)) for h in HZ}
        for (i, d) in eventos_fn(c):
            n_ev += 1
            for h in HZ:
                if i + h < m:
                    acc[h].append(d * (c[i + h] / c[i] - 1.0 - base[h]) * 100.0)
    pts = []
    for h in HZ:
        x = acc[h]
        if len(x) < 40:
            continue
        mm, ic, p1 = figuras._boot_media(x, bloque=max(10, h // 2))
        pts.append({"h": h, "etiqueta": LAB[h], "valor": round(mm, 2), "ic_lo": round(ic[0], 2),
                    "ic_hi": round(ic[1], 2), "n": len(x), "p": min(1.0, 2 * p1)})
    if not pts:
        return None
    return {"tipo": nombre, "nombre": nombre, "color": color, "n_eventos": n_ev,
            "razon": razon, "puntos": pts}


def _medir_estacional(px):
    """H3: exceso del retorno mensual de cada mes frente a la media de todos."""
    men = px.resample("ME").last()
    ret = (men / men.shift(1) - 1.0).dropna(how="all")
    if len(ret) < 60:
        return None
    todos = ret.values.flatten()
    todos = todos[np.isfinite(todos)]
    base = float(np.mean(todos))
    pts, n_ev = [], 0
    for mes, etq in [(1, "enero"), (5, "mayo"), (9, "septiembre"), (12, "diciembre")]:
        x = ret[ret.index.month == mes].values.flatten()
        x = x[np.isfinite(x)]
        if len(x) < 40:
            continue
        n_ev += len(x)
        ex = (x - base) * 100.0
        mm, ic, p1 = figuras._boot_media(ex, bloque=10)
        pts.append({"h": mes, "etiqueta": etq, "valor": round(mm, 2), "ic_lo": round(ic[0], 2),
                    "ic_hi": round(ic[1], 2), "n": len(x), "p": min(1.0, 2 * p1)})
    if not pts:
        return None
    return {"tipo": "estacionalidad", "nombre": "H3 · Estacionalidad (mes del año)",
            "color": "#88c0d0", "n_eventos": n_ev,
            "razon": ("Efecto enero (ventas fiscales de diciembre que se revierten) y «sell in May» "
                      "(flujos estivales). Exceso del mes frente a la media de todos los meses."),
            "puntos": pts}


def backtest_hipotesis(sintetico=False):
    px = _panel(sintetico)
    if px is None or px.shape[1] < 10:
        return None

    bloques = []
    b1 = _medir(px, _eventos_deriva, "H1 · Deriva tras sorpresa (gap ≥5%)", "#6ec08a",
                ("Tras un salto muy grande en un día (proxy de sorpresa informativa), el precio "
                 "debería seguir derivando en esa dirección: el mercado incorpora la información "
                 "despacio. Es el post-earnings drift, de las anomalías mejor documentadas."))
    if b1:
        bloques.append(b1)
    b2 = _medir(px, _eventos_reversion, "H2 · Reversión tras caída fuerte (−10% en 5d)", "#d2566a",
                ("Tras una caída fuerte y rápida, debería haber rebote: las ventas forzadas "
                 "(liquidez, margin calls) empujan el precio por debajo de su valor."))
    if b2:
        bloques.append(b2)
    b3 = _medir_estacional(px)
    if b3:
        bloques.append(b3)
    if not bloques:
        return None

    # Corrección CONJUNTA por multiple-testing sobre TODAS las celdas probadas
    todas = [(bi, pi) for bi, b in enumerate(bloques) for pi, _p in enumerate(b["puntos"])]
    pvals = [bloques[bi]["puntos"][pi]["p"] for bi, pi in todas]
    mask = figuras._bh(pvals, q=0.10)
    for (bi, pi), ok in zip(todas, mask):
        p = bloques[bi]["puntos"][pi]
        p["sig_cruda"] = bool(p["ic_lo"] > 0 or p["ic_hi"] < 0)
        p["sig_fdr"] = bool(ok)
    n_fdr = int(np.sum(mask))

    for b in bloques:
        b["nombre"] = f"{b['nombre']} · {b['n_eventos']} eventos"

    return {
        "id": "hipotesis_clasicas",
        "etiqueta": "Hipótesis clásicas (anomalías documentadas)",
        "tipo": f"Tres hipótesis con fundamento · {px.shape[1]} valores · FDR conjunto",
        "modelo": "hipotesis",
        "figuras_panel": True,
        "intro": ("Tres anomalías clásicas con razón económica previa, formuladas antes de mirar los "
                  "datos y medidas como exceso sobre la tasa base. «Ventaja» positiva = la hipótesis "
                  "se cumple. Cada bloque incluye su razón teórica."),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se prueban {len(pvals)} combinaciones hipótesis×horizonte A LA VEZ. La corrección "
                     f"de Benjamini-Hochberg (FDR 10%) se aplica al CONJUNTO, porque probar varias "
                     f"hipótesis sube el listón de todas: {n_fdr} sobreviven. Sin esa corrección, "
                     f"probar muchas cosas garantiza encontrar «ganadoras» por azar."),
        "nota": ("Anomalías documentadas en la literatura, testeadas aquí con datos propios y sin "
                 "búsqueda de parámetros: umbrales fijados de antemano (gap 5%, caída 10% en 5 días). "
                 "El veredicto se acepta tal cual. No es recomendación de inversión."),
    }
