"""
implied_vol.py — GARCH vs volatilidad IMPLÍCITA del mercado (VIX / GVZ).
=======================================================================
¿Tu GARCH bate al MERCADO, no solo al modelo ingenuo? El mercado ya pone precio a
la volatilidad futura vía la implícita (VIX para el S&P 500, GVZ para el oro).
Aquí comparamos, fuera de muestra:
  - implícita : lo que el mercado espera (VIX/GVZ, % anualizado)
  - GARCH     : lo que tu modelo prevé de volatilidad realizada
  - realizada : la volatilidad que de verdad ocurre después

Regla: si la implícita está muy por encima de tu previsión, la vol está "cara"
(sesgo a VENDER vol); si está por debajo, "barata" (sesgo a COMPRAR vol). El P&L a
vencimiento de una posición de vol ≈ (implícita − realizada), así que se mide sin
tocar opciones.

La pregunta honesta NO es si "vender vol siempre" gana (suele ganar: es la prima de
riesgo de varianza), sino si condicionar por el GARCH MEJORA sobre venderla siempre
a ciegas. Eso sería batir al mercado. Todo en papel. NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import data

ANNUAL = float(np.sqrt(252))
TRAIN = 504
REFIT = 63
H = 21   # horizonte de comparación (~1 mes, el plazo típico de la implícita)

CONFIGS = [
    {"id": "vol_implicita_sp500", "nombre": "S&P 500", "indice": "^VIX",
     "activo": "sp500", "color": "#5fb7c4"},
    {"id": "vol_implicita_oro", "nombre": "oro", "indice": "^GVZ",
     "activo": "oro", "color": "#e8b23a"},
]


def _cargar_indice(ticker):
    """Índice de volatilidad implícita (nivel = % anualizado)."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="max", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        return s if len(s) > 300 else None
    except Exception:
        return None


def _garch_forecast(ret_pct, train=TRAIN, refit=REFIT):
    """Walk-forward GARCH(1,1): vol anualizada (%) prevista a 1 día, sin lookahead."""
    from arch import arch_model
    vals = ret_pct.values
    n = len(vals)
    fc = np.full(n, np.nan)
    res = None
    for t in range(train, n):
        if res is None or (t - train) % refit == 0:
            try:
                res = arch_model(vals[:t], mean="Constant", vol="Garch",
                                 p=1, q=1, dist="t").fit(disp="off")
            except Exception:
                res = None
        if res is None:
            continue
        try:
            f = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
            fc[t] = np.sqrt(max(f, 1e-9)) * ANNUAL
        except Exception:
            pass
    return pd.Series(fc, index=ret_pct.index)


def _boot(x, n_boot=2000, bloque=21, seed=7):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 40:
        m = float(np.mean(x)) if n else 0.0
        return m, [round(m, 3), round(m, 3)], 1.0
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bloque)); base = np.arange(bloque)
    med = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n, size=nb)[:, None] + base[None, :]).ravel() % n
        med[i] = x[idx].mean()
    lo, hi = np.percentile(med, [5, 95]); m = float(np.mean(x))
    p = float(np.mean(med <= 0)) if m > 0 else float(np.mean(med >= 0))
    return m, [round(float(lo), 3), round(float(hi), 3)], round(min(1.0, 2 * p), 4)


def evaluar_vol_implicita(cfg, sintetico=False):
    activo = cfg["activo"]
    if sintetico:
        precio = data.cargar_sinteticos()["oro"]
        ret_tmp = (np.log(precio / precio.shift(1)).dropna()) * 100.0
        rv = ret_tmp.rolling(21).std() * ANNUAL
        rng = np.random.default_rng(3)
        implied = (rv.shift(-5).ffill().bfill() + 3.0
                   + rng.normal(0, 2.0, len(rv)))       # implícita ≈ realizada + prima + ruido
        implied = pd.Series(np.clip(implied.values, 5, 80), index=ret_tmp.index)
    else:
        precio = data.cargar_panel([activo])[activo]
        implied = _cargar_indice(cfg["indice"])
        if implied is None:
            return None

    ret = (np.log(precio / precio.shift(1)).dropna()) * 100.0
    prev = _garch_forecast(ret)
    impl = implied.reindex(ret.index).ffill()

    r = ret.values
    n = len(ret)
    realizada = np.full(n, np.nan)
    for t in range(n - H):
        realizada[t] = np.std(r[t + 1:t + 1 + H], ddof=1) * ANNUAL
    realizada = pd.Series(realizada, index=ret.index)

    df = pd.DataFrame({"impl": impl, "prev": prev, "real": realizada}).dropna()
    if len(df) < 250:
        return None

    vender_siempre = (df["impl"] - df["real"]).values          # cosecha la prima (VRP)
    caro = (df["impl"] > df["prev"]).values                     # señal del GARCH
    estrategia = np.where(caro, 1.0, -1.0) * vender_siempre     # vende si caro, compra si barato
    mejora = estrategia - vender_siempre                        # ¿aporta el GARCH?

    m_vs, _ic_vs, _ = _boot(vender_siempre)
    m_es, _ic_es, _ = _boot(estrategia)
    m_mj, ic_mj, p_mj = _boot(mejora)
    corr = float(np.corrcoef(df["impl"], df["prev"])[0, 1])
    prima = float(np.mean(df["impl"] - df["real"]))

    fechas = [d.strftime("%Y-%m-%d") for d in df.index]
    acum_es = np.cumsum(estrategia); acum_vs = np.cumsum(vender_siempre)
    PASO = max(1, len(fechas) // 180)
    curva = [{"fecha": fechas[i], "valor": round(float(acum_es[i]), 1)} for i in range(0, len(fechas), PASO)]
    curva2 = [{"fecha": fechas[i], "valor": round(float(acum_vs[i]), 1)} for i in range(0, len(fechas), PASO)]

    return {
        "id": cfg["id"],
        "etiqueta": f"Volatilidad implícita · {cfg['nombre']}",
        "tipo": f"GARCH vs implícita ({cfg['indice']}) · forward-test en papel · sin opciones",
        "modelo": "vol_implicita",
        "color": cfg["color"],
        "headline": {"valor": round(m_mj, 2),
                     "etiqueta": "Mejora del GARCH sobre «vender vol siempre»",
                     "sufijo": " pts", "decimales": 2},
        "significancia": {"p_valor": p_mj, "ic90": ic_mj,
                          "etiqueta": "mejora sobre vender siempre (pts de vol)"},
        "cards": [
            {"k": "Prima de riesgo (implícita − realizada)", "v": f"{prima:+.2f} pts", "tono": ""},
            {"k": "Vender vol siempre (media/op.)", "v": f"{m_vs:+.2f} pts", "tono": ""},
            {"k": "Estrategia GARCH (media/op.)", "v": f"{m_es:+.2f} pts", "tono": ""},
            {"k": "Correlación implícita-GARCH", "v": f"{corr:.2f}", "tono": ""},
            {"k": "Días evaluados", "v": str(len(df)), "tono": ""},
        ],
        "diagnostico": {},
        "curva": curva,
        "curva2": {"nombre": "Vender vol siempre (prima)", "datos": curva2},
        "curva_color": cfg["color"],
        "curva_unidad": " pts",
        "curva_base": 0.0,
        "curva_titulo": f"P&L acumulado en papel: GARCH vs vender vol siempre ({cfg['nombre']})",
        "curva_sub": ("La línea de color condiciona por el GARCH (vende vol cuando la implícita está "
                      "cara frente a la previsión, compra cuando está barata). La gris vende vol "
                      "siempre (cosecha la prima). Si la de color no supera a la gris, el GARCH no "
                      "aporta timing sobre lo que el mercado ya sabía. En papel; no es recomendación."),
    }


def evaluar_todas(sintetico=False):
    out = []
    for cfg in CONFIGS:
        try:
            r = evaluar_vol_implicita(cfg, sintetico=sintetico)
            if r:
                out.append(r)
        except Exception:
            pass
    return out
