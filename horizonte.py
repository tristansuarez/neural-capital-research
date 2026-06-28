"""
horizonte.py
============
Bloques de "ventaja segun el horizonte" para los modelos donde tiene sentido.

La pregunta es siempre la misma: ¿hasta que plazo sobrevive lo que el modelo
aporta? Donde el intervalo de confianza cruza el cero, se acabo la ventaja. El eje
cambia segun el modelo (exceso de retorno en el par y en KONCORDE; mejora de
prevision en GARCH), pero la lectura es identica.

NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _boot_media(x, n_boot=2000, bloque=10, seed=13):
    """Bootstrap por bloques de la media de x; devuelve (media, [ic_lo, ic_hi], p_valor)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max(2 * bloque, 30):
        m = float(np.mean(x)) if n else 0.0
        return m, [m, m], 1.0
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bloque))
    medias = np.empty(n_boot)
    base = np.arange(bloque)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + base[None, :]).ravel() % n
        medias[i] = x[idx].mean()
    lo, hi = np.percentile(medias, [5, 95])
    m = float(np.mean(x))
    p = float(np.mean(medias <= 0)) if m > 0 else float(np.mean(medias >= 0))
    return m, [float(lo), float(hi)], float(p)


# ======================= PAR ORO-PLATA (reversión) ======================= #
HZ_PAR = [5, 10, 21, 63, 126]
LAB_PAR = {5: "1 sem", 10: "2 sem", 21: "1 mes", 63: "3 meses", 126: "6 meses"}


def horizonte_par(panel, a="oro", b="plata", titulo_par="par", z_entry=1.5, lb=252, gap=5):
    """Tras una divergencia de >=z_entry sigmas, retorno de apostar a la reversión a h días."""
    df = panel.dropna()
    if not {a, b}.issubset(df.columns) or len(df) < lb + 200:
        return None
    lo = np.log(df[a].astype(float))
    lp = np.log(df[b].astype(float))
    cov = lo.rolling(lb).cov(lp)
    var = lp.rolling(lb).var()
    beta = cov / var                                   # ratio de cobertura rodante (sin lookahead)
    spread = lo - beta * lp
    z = (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std()
    s, zz = spread.values, z.values
    n = len(s)

    eventos, last = [], -10 ** 9
    for t in range(lb, n):
        if np.isfinite(zz[t]) and abs(zz[t]) >= z_entry and (t - last) >= gap:
            eventos.append(t); last = t

    puntos = []
    for h in HZ_PAR:
        pnl = []
        for t in eventos:
            if t + h < n and np.isfinite(s[t]) and np.isfinite(s[t + h]):
                # reversión: apostamos a que el spread vuelve hacia su media
                pnl.append(-np.sign(zz[t]) * (s[t + h] - s[t]) * 100.0)
        if len(pnl) < 20:
            continue
        m, ic, _ = _boot_media(pnl, bloque=max(5, h // 5))
        puntos.append({"h": h, "etiqueta": LAB_PAR[h], "valor": round(m, 2),
                       "ic_lo": round(ic[0], 2), "ic_hi": round(ic[1], 2), "n": len(pnl)})
    if not puntos:
        return None
    return {
        "titulo": f"Reversión del {titulo_par} según el horizonte",
        "sub": (f"Tras una divergencia de ≥1,5σ entre {a} y {b}, retorno medio de apostar a que el "
                f"spread vuelve a su media, a cada plazo. Si la barra no toca el cero, la reversión "
                f"aporta a ese horizonte."),
        "unidad": "%",
        "puntos": puntos,
        "nota": ("La reversión del spread suele ser lenta: a plazos cortos el movimiento puede "
                 "incluso continuar antes de revertir, y solo a horizontes largos el spread tiende "
                 "a volver. La curva muestra a qué plazo —si alguno— la reversión es distinguible "
                 "del azar."),
    }


# ======================= KONCORDE (event study) ======================= #
HZ_KON = [5, 10, 21, 42, 63]
LAB_KON = {5: "1 sem", 10: "2 sem", 21: "1 mes", 42: "2 meses", 63: "3 meses"}


def horizonte_koncorde(sintetico=False, n_tickers=40, anos=8):
    """Tras una señal de compra KONCORDE, exceso de retorno del valor vs su tasa base, a h días."""
    if sintetico:
        return None
    try:
        import datetime as dt
        import yfinance as yf
        import escaner_senales_telegram as esc
    except Exception:
        return None

    inicio = (dt.date.today() - dt.timedelta(days=int(anos * 365.25))).isoformat()
    try:
        tickers = esc.obtener_sp500()[:n_tickers]
    except Exception:
        return None
    try:
        regimen = esc.regimen_alcista(inicio)          # serie booleana (NVI del S&P)
    except Exception:
        regimen = None

    acc = {h: [] for h in HZ_KON}
    usados = 0
    for tk in tickers:
        try:
            df = yf.download(tk, start=inicio, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 300:
                continue
            k = esc.koncorde(df)
        except Exception:
            continue
        c = k["Close"].values
        marron, media = k["marron"].values, k["media"].values
        azul, verde = k["azul"].values, k["verde"].values
        fechas = k.index
        m = len(c)
        base = {h: np.nanmean(c[h:] / c[:-h] - 1.0) for h in HZ_KON}
        for i in range(1, m):
            if not (np.isfinite(marron[i]) and np.isfinite(media[i])):
                continue
            cruce = (marron[i] > media[i]) and (marron[i - 1] <= media[i - 1])
            if not (cruce and azul[i] > 0 and verde[i] > 0):
                continue
            if regimen is not None:
                try:
                    if not bool(regimen.reindex([fechas[i]], method="ffill").iloc[0]):
                        continue
                except Exception:
                    pass
            for h in HZ_KON:
                if i + h < m:
                    acc[h].append((c[i + h] / c[i] - 1.0 - base[h]) * 100.0)
        usados += 1

    puntos = []
    for h in HZ_KON:
        if len(acc[h]) < 50:
            continue
        mm, ic, _ = _boot_media(acc[h], bloque=max(5, h // 5))
        puntos.append({"h": h, "etiqueta": LAB_KON[h], "valor": round(mm, 2),
                       "ic_lo": round(ic[0], 2), "ic_hi": round(ic[1], 2), "n": len(acc[h])})
    if not puntos:
        return None
    return {
        "titulo": f"KONCORDE: exceso de retorno según el horizonte · backtest {anos} años",
        "sub": (f"Tras una señal de compra, retorno medio del valor frente a su tasa base, a cada "
                f"plazo ({usados} valores del S&P 500). Si la barra no toca el cero, la señal "
                f"anticipa retorno excedente a ese horizonte."),
        "unidad": "%",
        "puntos": puntos,
        "nota": ("Es un estudio de eventos histórico, distinto del forward-test en vivo de más "
                 "arriba. Confirma a Fosback: las señales de volumen a corto sobre acciones "
                 "individuales no producen exceso de retorno fiable; el intervalo abraza el cero."),
    }
