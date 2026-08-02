"""
pares_sectores.py — Reversión entre sectores cointegrados (ETFs líquidos).
=========================================================================
Hipótesis con fundamento económico, formulada antes de mirar:

  Sectores que comparten drivers macroeconómicos (tipos, ciclo, energía) mantienen
  una relación de largo plazo entre sus precios. Cuando divergen mucho, la relación
  tiende a restablecerse. A diferencia del par oro-plata, los ETFs sectoriales son
  MUY líquidos (spreads de 1-2 pb), así que la fricción no debería matar el efecto.

Método:
  1. Test de cointegración (Engle-Granger) sobre TODOS los pares del universo.
  2. Corrección por multiple-testing: probar N pares garantiza falsos cointegrados.
  3. Para los que cointegran, event study de la reversión tras divergencias >= 2σ,
     con bootstrap POR EPISODIO (mes) y sensibilidad al coste.

El veredicto se acepta tal cual. NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import figuras   # _boot_media, _bh

SECTORES = {
    "XLE": "Energía", "XLF": "Financiero", "XLK": "Tecnología", "XLV": "Salud",
    "XLI": "Industrial", "XLP": "Consumo básico", "XLY": "Consumo discrecional",
    "XLU": "Utilities", "XLB": "Materiales", "XLRE": "Inmobiliario",
    "XME": "Minería", "XOP": "Exploración petrolera", "KRE": "Banca regional",
    "SMH": "Semiconductores", "ITB": "Constructoras", "GDX": "Mineras de oro",
}
ANOS = 15
LB = 252        # ventana para z-score (causal)
Z_ENTRY = 2.0
GAP = 15
HZ = [10, 21, 63, 126]
LAB = {10: "2 sem", 21: "1 mes", 63: "3 meses", 126: "6 meses"}
COSTES = [0.0, 0.05, 0.10, 0.20, 0.40]


def _cargar(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(21)
        n = 2500
        idx = pd.bdate_range("2016-01-01", periods=n)
        base = np.cumsum(rng.normal(0.0003, 0.01, n))
        out = {}
        for i, tk in enumerate(list(SECTORES)[:8]):
            # dos primeros cointegrados por construcción, el resto independientes
            if i < 2:
                s = base + np.cumsum(rng.normal(0, 0.002, n)) * 0.3
            else:
                s = np.cumsum(rng.normal(0.0002, 0.011, n))
            out[tk] = 100 * np.exp(s)
        return pd.DataFrame(out, index=idx)
    import datetime as dt
    import yfinance as yf
    inicio = (dt.date.today() - dt.timedelta(days=int(ANOS * 365.25))).isoformat()
    cierres = {}
    tks = list(SECTORES)
    try:
        df = yf.download(tks, start=inicio, auto_adjust=True, progress=False,
                         group_by="ticker", threads=True)
    except Exception:
        return None
    for tk in tks:
        try:
            s = df[tk]["Close"].dropna()
            if len(s) > 1000:
                cierres[tk] = s
        except Exception:
            continue
    if len(cierres) < 6:
        return None
    # Alinea por fechas comunes pero sin dejar que un ETF de historia corta
    # (p. ej. XLRE, que nace en 2015) recorte todo el panel: descarta los que
    # reducirían demasiado la muestra.
    largos = sorted(cierres, key=lambda t: len(cierres[t]), reverse=True)
    ref = len(cierres[largos[0]])
    usar = [t for t in largos if len(cierres[t]) >= 0.7 * ref]
    if len(usar) < 6:
        usar = largos[:max(6, len(largos) // 2)]
    px = pd.DataFrame({t: cierres[t] for t in usar}).dropna()
    return px if len(px) > 800 else None


def _boot_ep(x, n_boot=3000, seed=17):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    k = len(x)
    if k < 8:
        return None
    rng = np.random.default_rng(seed)
    ms = np.array([x[rng.integers(0, k, k)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(ms, [5, 95]); m = float(np.mean(x))
    p = float(np.mean(ms <= 0)) if m > 0 else float(np.mean(ms >= 0))
    return {"n": k, "media": round(m, 2), "ic": [round(float(lo), 2), round(float(hi), 2)],
            "p": round(min(1.0, 2 * p), 4)}


def evaluar_pares_sectores(sintetico=False):
    from statsmodels.tsa.stattools import coint

    px = _cargar(sintetico)
    if px is None or px.shape[1] < 6:
        return None
    tks = list(px.columns)
    fechas = [d.strftime("%Y-%m") for d in px.index]

    # 1) cointegración de todos los pares + FDR
    pares, pv = [], []
    for i in range(len(tks)):
        for j in range(i + 1, len(tks)):
            a, b = tks[i], tks[j]
            try:
                _s, p, _c = coint(np.log(px[a]), np.log(px[b]))
            except Exception:
                continue
            pares.append((a, b)); pv.append(float(p))
    if not pares:
        return None
    mask = figuras._bh(pv, q=0.10)
    cointegrados = [(pares[k], pv[k]) for k in range(len(pares)) if mask[k]]
    mejores = sorted(zip(pares, pv), key=lambda x: x[1])[:8]

    def _sin_resultado(motivo):
        """Un «no cointegra nada» es un resultado válido: se publica, no se oculta."""
        lst = ", ".join(f"{SECTORES.get(a, a)}–{SECTORES.get(b, b)} (p={p:.3f})"
                        for (a, b), p in mejores)
        return {
            "id": "pares_sectores",
            "etiqueta": "Pares sectoriales (cointegración)",
            "tipo": f"{len(pares)} pares probados · {len(cointegrados)} cointegran tras FDR",
            "modelo": "pares_sectores",
            "sin_datos": True,
            "intro": motivo,
            "nota": (f"Se probaron {len(pares)} pares de ETFs sectoriales. Tras corregir por "
                     f"multiple-testing (Benjamini-Hochberg, FDR 10%), cointegran {len(cointegrados)}. "
                     f"Pares con menor p-valor: {lst}. "
                     f"Que casi ningún par de sectores mantenga una relación estacionaria estable en "
                     f"15 años es en sí un resultado: las relaciones entre sectores se rompen con los "
                     f"cambios de ciclo y de composición. No es recomendación de inversión."),
        }

    if not cointegrados:
        return _sin_resultado("Ningún par de sectores supera el test de cointegración tras la "
                              "corrección por multiple-testing.")

    # 2) reversión de los pares que cointegran (agregado, por episodio)
    acc = {h: [] for h in HZ}
    por_mes = {h: {} for h in HZ}
    n_ev = 0
    for (a, b), _p in cointegrados:
        la, lb_ = np.log(px[a].values), np.log(px[b].values)
        cov = pd.Series(la).rolling(LB).cov(pd.Series(lb_))
        var = pd.Series(lb_).rolling(LB).var()
        beta = (cov / var).values
        spread = la - beta * lb_
        s = pd.Series(spread)
        z = ((s - s.rolling(LB).mean()) / s.rolling(LB).std()).values
        n = len(z); last = -10 ** 9
        for t in range(LB, n):
            if np.isfinite(z[t]) and abs(z[t]) >= Z_ENTRY and (t - last) >= GAP:
                last = t; n_ev += 1
                bt = beta[t]                      # beta CONGELADA en la entrada (lo real al operar)
                if not np.isfinite(bt):
                    continue
                for h in HZ:
                    if t + h < n:
                        # P&L de la posición abierta en t, valorada a t+h con la misma beta
                        var_spread = (la[t + h] - bt * lb_[t + h]) - (la[t] - bt * lb_[t])
                        pnl = -np.sign(z[t]) * var_spread * 100.0
                        # normaliza por el apalancamiento bruto (|1| + |beta|)
                        pnl = pnl / (1.0 + abs(bt))
                        if np.isfinite(pnl):
                            acc[h].append(pnl)
                            por_mes[h].setdefault(fechas[t], []).append(pnl)

    puntos = []
    for h in HZ:
        x = acc[h]
        if len(x) < 30:
            continue
        m, ic, p1 = figuras._boot_media(x, bloque=max(10, h // 2))
        ep = np.array([float(np.mean(v)) for v in por_mes[h].values()])
        rep = _boot_ep(ep)
        puntos.append({"h": h, "etiqueta": LAB[h], "valor": round(m, 2),
                       "ic_lo": round(ic[0], 2), "ic_hi": round(ic[1], 2), "n": len(x),
                       "p": min(1.0, 2 * p1),
                       "ep_media": rep["media"] if rep else None,
                       "ep_p": rep["p"] if rep else None,
                       "ep_n": rep["n"] if rep else None})
    if not puntos:
        return _sin_resultado("Hay pares cointegrados, pero no se acumulan suficientes eventos de "
                              "divergencia (≥2σ) para medir la reversión con fiabilidad.")

    mask2 = figuras._bh([p["p"] for p in puntos], q=0.10)
    for p, ok in zip(puntos, mask2):
        p["sig_cruda"] = bool(p["ic_lo"] > 0 or p["ic_hi"] < 0)
        p["sig_fdr"] = bool(ok)
    n_fdr = int(np.sum(mask2))

    # sensibilidad al coste sobre el horizonte más largo con datos
    hmax = puntos[-1]["h"]
    ep_max = np.array([float(np.mean(v)) for v in por_mes[hmax].values()])
    sens = []
    for c in COSTES:
        r = _boot_ep(ep_max - c)
        if r:
            sens.append((c, r["media"], r["p"]))
    sens_txt = "".join(
        f"<tr><td>{c:.2f}%</td><td class='{'pos' if m > 0 else 'neg'}'>{m:+.2f}%</td>"
        f"<td class='{'pos' if p <= 0.10 else 'est-obs'}'>{p}</td></tr>" for c, m, p in sens)

    lista = ", ".join(f"{SECTORES.get(a, a)}–{SECTORES.get(b, b)} (p={p:.3f})"
                      for (a, b), p in cointegrados[:12]) or "ninguno"

    bloque = {
        "tipo": "reversion_sectores", "nombre": "Reversión del spread entre sectores cointegrados",
        "color": "#5fb7c4", "n_eventos": n_ev,
        "puntos": [{"etiqueta": p["etiqueta"], "valor": p["valor"], "ic_lo": p["ic_lo"],
                    "ic_hi": p["ic_hi"], "n": p["n"], "sig_cruda": p["sig_cruda"],
                    "sig_fdr": p["sig_fdr"]} for p in puntos],
    }
    ep_txt = "".join(
        f"<tr><td>{p['etiqueta']}</td><td>{p['valor']:+.2f}%</td>"
        f"<td class='{'pos' if (p['ep_media'] or 0) > 0 else 'neg'}'>{p['ep_media']:+.2f}%</td>"
        f"<td class='{'pos' if (p['ep_p'] or 1) <= 0.10 else 'est-obs'}'>{p['ep_p']}</td>"
        f"<td class='est-obs'>{p['ep_n']}</td></tr>" for p in puntos)

    return {
        "id": "pares_sectores",
        "etiqueta": "Pares sectoriales (cointegración)",
        "tipo": f"{len(pares)} pares probados · {len(cointegrados)} cointegran tras FDR · ETFs líquidos",
        "modelo": "pares_sectores",
        "figuras_panel": True,
        "intro": ("Sectores que comparten drivers macro mantienen una relación de largo plazo; cuando "
                  "divergen, debería restablecerse. A diferencia de los metales, los ETFs sectoriales "
                  "son muy líquidos, así que la fricción no debería matar el efecto. «Ventaja» positiva "
                  "= la reversión aporta."),
        "figuras": [bloque],
        "n_celdas": len(puntos),
        "n_fdr": n_fdr,
        "nota_fdr": (f"Se probaron {len(pares)} pares de sectores para cointegración. Probar tantos "
                     f"garantiza falsos positivos, así que se aplica Benjamini-Hochberg: sobreviven "
                     f"{len(cointegrados)}. Pares cointegrados: {lista}."),
        "nota": ("<b>Por episodio y sensibilidad al coste</b> (lo que decide si es operable):"
                 "<div class='ops-scroll'><table class='ops'><thead><tr><th>Horizonte</th>"
                 "<th>Por evento</th><th>Por episodio (mes)</th><th>p episodio</th><th>episodios</th>"
                 f"</tr></thead><tbody>{ep_txt}</tbody></table></div>"
                 "<br><b>Sensibilidad al coste</b> (horizonte más largo, por episodio):"
                 "<div class='ops-scroll'><table class='ops'><thead><tr><th>Coste</th>"
                 f"<th>Neto</th><th>p</th></tr></thead><tbody>{sens_txt}</tbody></table></div>"
                 "<br>Los ETFs sectoriales tienen spreads de 1-2 puntos básicos, así que aquí los "
                 "costes realistas están en la parte baja de la tabla. No es recomendación de inversión."),
    }
