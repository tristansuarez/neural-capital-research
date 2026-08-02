"""
hipotesis2.py — Cinco hipótesis nuevas con fundamento, testeadas a la vez.
=========================================================================
Formuladas ANTES de mirar los datos, cada una con su razón económica:

  H4. MOMENTUM DE SERIE TEMPORAL (índice). Estar dentro del mercado cuando está
      sobre su media de 10 meses, fuera cuando está debajo. Razón: los inversores
      reaccionan tarde a los cambios de tendencia. Su promesa NO es más retorno,
      sino MENOS caída máxima: se juzga por retorno ajustado a riesgo.

  H5. BAJA VOLATILIDAD. Las acciones menos volátiles rinden más ajustado a riesgo.
      Razón: quien no puede apalancarse sobrepaga por las volátiles buscando
      retorno, abaratando las tranquilas.

  H6. REVERSIÓN MENSUAL CROSS-SECTIONAL. Comprar los peores del mes pasado y
      vender los mejores. Razón: presión de liquidez a fin de mes que se revierte.

  H7. TAMAÑO (proxy por precio bajo dentro del índice). Los valores más pequeños
      rinden más. Razón: prima de riesgo y menor cobertura de analistas.

  H8. INCLUSIÓN EN EL ÍNDICE. Al entrar una acción en el S&P 500, los fondos
      indexados deben comprarla: demanda forzada y predecible.

Todas se miden fuera de muestra, con bootstrap POR EPISODIO (mes) donde procede y
corrección FDR CONJUNTA sobre todas las celdas. El veredicto se acepta tal cual.
NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras

ANOS = 20
N_TICKERS = 100
COSTE_MES = 0.05      # coste por rebalanceo mensual de una cartera de ETFs/acciones líquidas


def _panel(sintetico=False, n_tickers=N_TICKERS):
    if sintetico:
        rng = np.random.default_rng(31)
        n, k = 3000, 40
        idx = pd.bdate_range("2014-01-01", periods=n)
        r = rng.normal(0.0003, 0.014, (n, k))
        return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                            columns=[f"SYN{i}" for i in range(k)])
    import datetime as dt
    import yfinance as yf
    import escaner_senales_telegram as esc
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    tickers = esc.obtener_sp500()[:n_tickers]
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
                if len(s) > 1200:
                    cierres[tk] = s
            except Exception:
                continue
    if len(cierres) < 20:
        return None
    return pd.DataFrame(cierres).ffill()


def _boot(x, seed=23, n_boot=3000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    k = len(x)
    if k < 12:
        return None
    rng = np.random.default_rng(seed)
    ms = np.array([x[rng.integers(0, k, k)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(ms, [5, 95]); m = float(np.mean(x))
    p = float(np.mean(ms <= 0)) if m > 0 else float(np.mean(ms >= 0))
    return {"n": k, "m": round(m, 3), "ic": [round(float(lo), 3), round(float(hi), 3)],
            "p": round(min(1.0, 2 * p), 4)}


def _mensual(px):
    return px.resample("ME").last()


def h4_momentum_indice(px):
    """Dentro/fuera del índice según su media de 10 meses. Mide exceso ajustado a riesgo."""
    idx = px.mean(axis=1)
    men = idx.resample("ME").last()
    ret = men.pct_change()
    ma = men.rolling(10).mean()
    dentro = (men.shift(1) > ma.shift(1))          # decisión con info del mes anterior
    est = ret.where(dentro, 0.0) - np.where(dentro, COSTE_MES / 100.0, 0.0)
    bh = ret
    df = pd.DataFrame({"est": est, "bh": bh}).dropna()
    if len(df) < 60:
        return None
    e, b = df["est"].values, df["bh"].values
    sh_e = float(np.mean(e) / np.std(e, ddof=1) * np.sqrt(12)) if np.std(e) > 0 else 0.0
    sh_b = float(np.mean(b) / np.std(b, ddof=1) * np.sqrt(12)) if np.std(b) > 0 else 0.0
    dd_e = float((np.cumprod(1 + e) / np.maximum.accumulate(np.cumprod(1 + e)) - 1).min() * 100)
    dd_b = float((np.cumprod(1 + b) / np.maximum.accumulate(np.cumprod(1 + b)) - 1).min() * 100)
    r = _boot((e - b) * 100)
    return {"nombre": "H4 · Momentum del índice (media 10 meses)", "color": "#6ec08a",
            "razon": ("Estar dentro cuando el índice está sobre su media de 10 meses. Su promesa no es "
                      "más retorno sino menos caída: se juzga por Sharpe y drawdown."),
            "n_eventos": len(df),
            "puntos": [{"etiqueta": "exceso mensual", "valor": r["m"], "ic_lo": r["ic"][0],
                        "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": (f"Sharpe estrategia {sh_e:.2f} vs comprar y mantener {sh_b:.2f}. "
                      f"Caída máxima {dd_e:.1f}% vs {dd_b:.1f}%.")}


def h5_baja_volatilidad(px):
    """Quintil de menor volatilidad (252d) vs quintil de mayor, rebalanceo mensual."""
    men = _mensual(px)
    ret = men.pct_change()
    vol = px.pct_change().rolling(252).std().resample("ME").last()
    filas_lo, filas_hi = [], []
    for i in range(1, len(men)):
        v = vol.iloc[i - 1]                              # info del mes anterior
        r = ret.iloc[i]
        m = v.notna() & r.notna()
        if m.sum() < 20:
            continue
        q = v[m].quantile([0.2, 0.8])
        lo = r[m][v[m] <= q.iloc[0]].mean()
        hi = r[m][v[m] >= q.iloc[1]].mean()
        if np.isfinite(lo) and np.isfinite(hi):
            filas_lo.append(lo); filas_hi.append(hi)
    if len(filas_lo) < 40:
        return None
    lo = np.array(filas_lo); hi = np.array(filas_hi)
    dif = (lo - hi) * 100 - COSTE_MES
    r = _boot(dif)
    sh_lo = float(np.mean(lo) / np.std(lo, ddof=1) * np.sqrt(12))
    sh_hi = float(np.mean(hi) / np.std(hi, ddof=1) * np.sqrt(12))
    return {"nombre": "H5 · Baja volatilidad (quintil bajo − quintil alto)", "color": "#5fb7c4",
            "razon": ("Quien no puede apalancarse sobrepaga por las acciones volátiles, abaratando "
                      "las tranquilas. Se compara el quintil menos volátil contra el más volátil."),
            "n_eventos": len(lo),
            "puntos": [{"etiqueta": "exceso mensual", "valor": r["m"], "ic_lo": r["ic"][0],
                        "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": f"Sharpe quintil bajo {sh_lo:.2f} vs quintil alto {sh_hi:.2f}."}


def h6_reversion_mensual(px):
    """Comprar los peores del mes anterior, vender los mejores (cross-sectional)."""
    men = _mensual(px)
    ret = men.pct_change()
    dif = []
    for i in range(2, len(ret)):
        prev, cur = ret.iloc[i - 1], ret.iloc[i]
        m = prev.notna() & cur.notna()
        if m.sum() < 20:
            continue
        q = prev[m].quantile([0.2, 0.8])
        peor = cur[m][prev[m] <= q.iloc[0]].mean()
        mejor = cur[m][prev[m] >= q.iloc[1]].mean()
        if np.isfinite(peor) and np.isfinite(mejor):
            dif.append((peor - mejor) * 100 - COSTE_MES)
    if len(dif) < 40:
        return None
    r = _boot(np.array(dif))
    return {"nombre": "H6 · Reversión mensual (peores − mejores)", "color": "#d2566a",
            "razon": ("Presión de liquidez a fin de mes que se revierte: los más castigados rebotan "
                      "frente a los más subidos. Es relativa, no absoluta como H2."),
            "n_eventos": len(dif),
            "puntos": [{"etiqueta": "exceso mensual", "valor": r["m"], "ic_lo": r["ic"][0],
                        "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": ""}


def h7_tamano(px):
    """Proxy de tamaño por precio: quintil de menor precio vs mayor (dentro del índice)."""
    men = _mensual(px)
    ret = men.pct_change()
    dif = []
    for i in range(1, len(men)):
        p = men.iloc[i - 1]; r = ret.iloc[i]
        m = p.notna() & r.notna()
        if m.sum() < 20:
            continue
        q = p[m].quantile([0.2, 0.8])
        peq = r[m][p[m] <= q.iloc[0]].mean()
        gra = r[m][p[m] >= q.iloc[1]].mean()
        if np.isfinite(peq) and np.isfinite(gra):
            dif.append((peq - gra) * 100 - COSTE_MES)
    if len(dif) < 40:
        return None
    r = _boot(np.array(dif))
    return {"nombre": "H7 · Tamaño (proxy: precio bajo − precio alto)", "color": "#b48ad6",
            "razon": ("Los valores más pequeños rinden más por prima de riesgo y menor cobertura. "
                      "Proxy imperfecto: usamos el precio como aproximación al tamaño, porque no "
                      "disponemos de capitalización histórica."),
            "n_eventos": len(dif),
            "puntos": [{"etiqueta": "exceso mensual", "valor": r["m"], "ic_lo": r["ic"][0],
                        "ic_hi": r["ic"][1], "n": r["n"], "p": r["p"]}],
            "extra": "Ojo: el precio nominal es un proxy pobre del tamaño; tómese con cautela."}


def backtest_hipotesis2(sintetico=False):
    px = _panel(sintetico)
    if px is None or px.shape[1] < 20:
        return None

    bloques = []
    for fn in (h4_momentum_indice, h5_baja_volatilidad, h6_reversion_mensual, h7_tamano):
        try:
            b = fn(px)
        except Exception:
            b = None
        if b:
            bloques.append(b)
    if not bloques:
        return None

    # H8 (inclusión en el índice) requiere fechas de alta que no están en los precios
    nota_h8 = ("H8 (inclusión en el índice) no se ha podido testear: exige la fecha de entrada de "
               "cada valor al S&P 500, dato que no está en las series de precio y que Wikipedia "
               "ofrece de forma incompleta. Se declara no evaluada en lugar de aproximarla mal.")

    pvals = [b["puntos"][0]["p"] for b in bloques]
    mask = figuras._bh(pvals, q=0.10)
    for b, ok in zip(bloques, mask):
        p = b["puntos"][0]
        p["sig_cruda"] = bool(p["ic_lo"] > 0 or p["ic_hi"] < 0)
        p["sig_fdr"] = bool(ok)
    n_fdr = int(np.sum(mask))

    detalle = "".join(
        f"<div class='vf-row'><span class='dot' style='background:{b['color']}'></span>"
        f"<b>{b['nombre']}</b></div><div class='ch-sub' style='margin:4px 0 12px 18px'>"
        f"{b['razon']}{(' ' + b['extra']) if b.get('extra') else ''}</div>"
        for b in bloques)

    for b in bloques:
        b["tipo"] = b["nombre"]
        b["nombre"] = f"{b['nombre']} · {b['n_eventos']} meses"

    return {
        "id": "hipotesis_nuevas",
        "etiqueta": "Hipótesis nuevas (momentum, baja vol, reversión, tamaño)",
        "tipo": f"Cuatro hipótesis con fundamento · {px.shape[1]} valores · FDR conjunto · neto de costes",
        "modelo": "hipotesis2",
        "figuras_panel": True,
        "intro": ("Cuatro anomalías con razón económica previa, formuladas antes de mirar los datos, "
                  "con rebalanceo mensual y coste aplicado. «Ventaja» positiva = la hipótesis se "
                  "cumple. Cada una se explica abajo." + detalle),
        "figuras": bloques,
        "n_celdas": len(pvals),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se prueban {len(pvals)} hipótesis A LA VEZ y la corrección de Benjamini-Hochberg "
                     f"(FDR 10%) se aplica al conjunto: {n_fdr} sobreviven. Probar varias sube el "
                     f"listón de todas; sin corregir, probar muchas garantiza «ganadoras» por azar."),
        "nota": ("Umbrales fijados de antemano (quintiles, media de 10 meses) y coste de "
                 f"{COSTE_MES}% por rebalanceo mensual. El veredicto se acepta tal cual. " + nota_h8 +
                 " No es recomendación de inversión."),
    }
