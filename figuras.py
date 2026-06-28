"""
figuras.py — Detección de figuras técnicas + backtest honesto (event study).
============================================================================
Recorre una muestra del S&P 500 (velas diarias), detecta figuras de chartismo con
REGLAS FIJAS (pivotes, pendientes, tolerancias) y, para cada tipo, mide fuera de
muestra el retorno tras la ruptura frente a la tasa base del propio valor, a varios
horizontes. Igual que el event-study de KONCORDE, pero con figuras de gráfico.

Honestidad por delante:
  - Reglas geométricas fijas y reproducibles, no "ojo" subjetivo. No coincidirán
    exactamente con lo que dibujaría una persona, pero son objetivas y sin sesgo.
  - Cada figura se mide como EXCESO sobre la tasa base del valor, y orientada a su
    dirección: una figura alcista "funciona" si el precio sube; una bajista, si baja.
  - Se corrige por MULTIPLE-TESTING (Benjamini-Hochberg, FDR): se prueban muchas
    figuras x horizontes a la vez, así que algo "sale significativo" solo por azar.
    Sin esa corrección, el resultado engaña.
  - La literatura y nuestro propio KONCORDE anticipan que casi ninguna figura tendrá
    edge fiable. Esto lo MIDE, no lo promete.

NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HZ_FIG = [5, 10, 21, 42, 63]
LAB_FIG = {5: "1 sem", 10: "2 sem", 21: "1 mes", 42: "2 meses", 63: "3 meses"}

# Horizontes intradía (en velas de 1 hora; ~6,5 velas por sesión)
HZ_FIG_INTRA = [3, 7, 14, 35, 70]
LAB_FIG_INTRA = {3: "½ día", 7: "1 día", 14: "2 días", 35: "1 sem", 70: "2 sem"}

# tipo -> (nombre legible, dirección por defecto, color, sesgo)
FIGURAS = {
    "ruptura_resistencia":       ("Ruptura de resistencia", +1, "#5fb7c4"),
    "ruptura_soporte":           ("Ruptura de soporte",     -1, "#d2566a"),
    "ruptura_tendencia_bajista": ("Ruptura de línea bajista (al alza)", +1, "#6ec08a"),
    "ruptura_tendencia_alcista": ("Ruptura de línea alcista (a la baja)", -1, "#d08a5a"),
    "doble_suelo":               ("Doble suelo", +1, "#b48ad6"),
    "doble_techo":               ("Doble techo", -1, "#e8b23a"),
}


# --------------------------- detección ---------------------------
def detectar(h, l, c, k=5, tol=0.03, win=60, buf=0.005):
    """Devuelve lista de eventos (idx_ruptura, tipo, direccion)."""
    n = len(c)
    if n < 3 * k + 10:
        return []
    ph = [i for i in range(k, n - k) if h[i] == max(h[i - k:i + k + 1])]
    pl = [i for i in range(k, n - k) if l[i] == min(l[i - k:i + k + 1])]
    ev = []

    # Doble techo / doble suelo (dos pivotes del mismo tipo a precio similar + ruptura del cuello)
    for a, b in zip(ph, ph[1:]):
        if 10 <= b - a <= win and abs(h[a] - h[b]) / h[a] < tol:
            cuello = float(l[a:b + 1].min())
            for j in range(b + 1, min(b + win, n)):
                if c[j] < cuello * (1 - buf):
                    ev.append((j, "doble_techo", -1)); break
    for a, b in zip(pl, pl[1:]):
        if 10 <= b - a <= win and abs(l[a] - l[b]) / l[a] < tol:
            cuello = float(h[a:b + 1].max())
            for j in range(b + 1, min(b + win, n)):
                if c[j] > cuello * (1 + buf):
                    ev.append((j, "doble_suelo", +1)); break

    # Líneas de tendencia / resistencia y soporte (par de pivotes -> recta -> ruptura)
    for a, b in zip(ph, ph[1:]):
        if not (5 <= b - a <= win):
            continue
        slope = (h[b] - h[a]) / (b - a)
        for j in range(b + 1, min(b + win, n)):
            linea = h[b] + slope * (j - b)
            if c[j] > linea * (1 + buf):
                tipo = "ruptura_tendencia_bajista" if slope < 0 else "ruptura_resistencia"
                ev.append((j, tipo, +1)); break
    for a, b in zip(pl, pl[1:]):
        if not (5 <= b - a <= win):
            continue
        slope = (l[b] - l[a]) / (b - a)
        for j in range(b + 1, min(b + win, n)):
            linea = l[b] + slope * (j - b)
            if c[j] < linea * (1 - buf):
                tipo = "ruptura_tendencia_alcista" if slope > 0 else "ruptura_soporte"
                ev.append((j, tipo, -1)); break
    return ev


# --------------------------- estadística ---------------------------
def _boot_media(x, n_boot=2000, bloque=10, seed=7):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < max(2 * bloque, 30):
        m = float(np.mean(x)) if n else 0.0
        return m, [m, m], 1.0
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bloque)); base = np.arange(bloque)
    med = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n, size=nb)[:, None] + base[None, :]).ravel() % n
        med[i] = x[idx].mean()
    lo, hi = np.percentile(med, [5, 95]); m = float(np.mean(x))
    p1 = float(np.mean(med <= 0)) if m > 0 else float(np.mean(med >= 0))
    return m, [float(lo), float(hi)], p1


def _bh(pvals, q=0.10):
    """Benjamini-Hochberg: devuelve máscara de los que sobreviven al FDR q."""
    p = np.asarray(pvals, float); m = len(p)
    if m == 0:
        return np.array([], bool)
    orden = np.argsort(p)
    umbral = q * (np.arange(1, m + 1) / m)
    pasa_ord = p[orden] <= umbral
    k = np.where(pasa_ord)[0]
    mask = np.zeros(m, bool)
    if len(k):
        corte = orden[:k.max() + 1]
        mask[corte] = True
    return mask


# --------------------------- datos ---------------------------
def _ohlc_sintetico(seed, n=2600):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.012, n)
    c = 100 * np.exp(np.cumsum(r))
    ruido = np.abs(rng.normal(0, 0.006, n)) * c
    h = c + ruido; l = c - ruido
    return h, l, c


def _muestra_sintetica(n_tickers=25):
    for s in range(n_tickers):
        yield f"SYN{s}", _ohlc_sintetico(1000 + s)


def _muestra_real(n_tickers, anos):
    import datetime as dt
    import yfinance as yf
    import escaner_senales_telegram as esc
    inicio = (dt.date.today() - dt.timedelta(days=int(anos * 365.25))).isoformat()
    tickers = esc.obtener_sp500()[:n_tickers]
    for tk in tickers:
        try:
            df = yf.download(tk, start=inicio, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 300:
                continue
            yield tk, (df["High"].values, df["Low"].values, df["Close"].values)
        except Exception:
            continue


def _muestra_intradia(n_tickers=120, period="2y"):
    """Velas de 1 hora por lotes (Yahoo da ~730 días de histórico horario)."""
    import yfinance as yf
    import escaner_senales_telegram as esc
    tickers = esc.obtener_sp500()[:n_tickers]
    for j in range(0, len(tickers), 40):
        chunk = tickers[j:j + 40]
        try:
            df = yf.download(chunk, period=period, interval="1h", group_by="ticker",
                             auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
        for tk in chunk:
            try:
                sub = df[tk].dropna()
                if len(sub) < 200:
                    continue
                yield tk, (sub["High"].values, sub["Low"].values, sub["Close"].values)
            except Exception:
                continue


# --------------------------- backtest ---------------------------
def backtest_figuras(sintetico=False, n_tickers=60, anos=10, intradia=False):
    if intradia:
        hz, lab = HZ_FIG_INTRA, LAB_FIG_INTRA
        fuente = _muestra_sintetica(n_tickers=20) if sintetico else _muestra_intradia()
        id_exp, etiqueta = "figuras_intradia", "Figuras técnicas · intradía (1h)"
        tipo_txt_fmt = "Chartismo intradía · velas 1h · event study · {u} valores · ~2 años"
        intro_extra = (" Detección sobre velas de 1 hora; la ventaja se mide a horizontes intradía "
                       "y de pocos días. El histórico horario gratuito llega a ~2 años.")
    else:
        hz, lab = HZ_FIG, LAB_FIG
        fuente = _muestra_sintetica() if sintetico else _muestra_real(n_tickers, anos)
        id_exp, etiqueta = "figuras_tecnicas", "Figuras técnicas (S&P 500)"
        tipo_txt_fmt = "Chartismo · event study · {u} valores · backtest %d años" % anos
        intro_extra = ""

    acc = {tipo: {h: [] for h in hz} for tipo in FIGURAS}
    n_eventos = {tipo: 0 for tipo in FIGURAS}
    tickers_usados = []
    try:
        for _tk, (H, L, C) in fuente:
            try:
                ev = detectar(np.asarray(H, float), np.asarray(L, float), np.asarray(C, float))
            except Exception:
                continue
            m = len(C)
            base = {h: float(np.nanmean(C[h:] / C[:-h] - 1.0)) for h in hz}
            for (i, tipo, d) in ev:
                n_eventos[tipo] += 1
                for h in hz:
                    if i + h < m:
                        acc[tipo][h].append(d * (C[i + h] / C[i] - 1.0 - base[h]) * 100.0)
            tickers_usados.append(_tk)
    except Exception:
        return None
    usados = len(tickers_usados)

    # estadística por celda (tipo x horizonte) + recogida de p-valores para el FDR
    celdas = []
    for tipo in FIGURAS:
        for h in hz:
            x = acc[tipo][h]
            if len(x) < 40:
                continue
            mm, ic, p1 = _boot_media(x, bloque=max(5, h // 5))
            p2 = min(1.0, 2 * p1)                  # dos colas
            celdas.append({"tipo": tipo, "h": h, "valor": round(mm, 2),
                           "ic_lo": round(ic[0], 2), "ic_hi": round(ic[1], 2),
                           "n": len(x), "p": p2})
    if not celdas:
        return None
    mask = _bh([c["p"] for c in celdas], q=0.10)
    for c, ok in zip(celdas, mask):
        c["sig_cruda"] = bool(c["ic_lo"] > 0 or c["ic_hi"] < 0)
        c["sig_fdr"] = bool(ok)

    figuras = []
    for tipo, (nombre, _dir, color) in FIGURAS.items():
        pts = [c for c in celdas if c["tipo"] == tipo]
        if not pts:
            continue
        figuras.append({
            "tipo": tipo, "nombre": nombre, "color": color,
            "n_eventos": n_eventos[tipo],
            "puntos": [{"etiqueta": lab[c["h"]], "valor": c["valor"],
                        "ic_lo": c["ic_lo"], "ic_hi": c["ic_hi"], "n": c["n"],
                        "sig_cruda": c["sig_cruda"], "sig_fdr": c["sig_fdr"]} for c in pts],
        })
    if not figuras:
        return None
    n_fdr = sum(1 for c in celdas if c["sig_fdr"])
    return {
        "id": id_exp,
        "etiqueta": etiqueta,
        "tipo": tipo_txt_fmt.format(u=usados),
        "modelo": "figuras",
        "figuras_panel": True,
        "tickers": sorted(tickers_usados),
        "intro": ("Cada figura se detecta con reglas fijas y se mide su retorno tras la ruptura, "
                  "orientado a su dirección (alcista = sube; bajista = baja), frente a la tasa base "
                  "del valor. «Ventaja» positiva = la figura funciona en su sentido." + intro_extra),
        "figuras": figuras,
        "n_celdas": len(celdas),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se han evaluado {len(celdas)} combinaciones figura×horizonte. Tras corregir "
                     f"por multiple-testing (Benjamini-Hochberg, FDR 10%), {n_fdr} sobreviven. "
                     f"La columna «cruda» marca lo que parece significativo sin corregir; la columna "
                     f"«FDR» marca lo que aguanta de verdad. La diferencia es la trampa del data mining."),
        "nota": ("Estudio de eventos histórico con reglas reproducibles. Si casi todo «abraza el cero» "
                 "tras la corrección, es coherente con que el chartismo no dé edge fiable. "
                 "No es recomendación de inversión."),
    }
