"""
ml_forward.py — Machine learning (gradient boosting) con walk-forward estricto.
==============================================================================
PROTOCOLO FIJADO DE ANTEMANO (antes de mirar ningún resultado):

  - Modelo   : HistGradientBoosting sobre features de precio estándar.
  - Features : momentum 21/63/126/252d, volatilidad 21/63d, distancia a máximo de
               252d, RSI(14), retorno 5d. Todas causales (solo pasado).
  - Objetivo : retorno relativo del valor frente a la media de la sección cruzada
               en los siguientes 21 días.
  - Validación: WALK-FORWARD estricto. Entrena con datos hasta el día t, predice
               t+1..t+21, refit cada 21 días. Nunca ve el futuro.
  - Cartera  : largo del quintil superior de la predicción, equiponderado.
  - Métrica  : retorno NETO de costes (COST_BPS por rotación) frente a comprar y
               mantener el mismo universo, y bootstrap sobre periodos.
  - Veredicto: se acepta el resultado, gane o pierda. UNA configuración, no cien.

Motivo del protocolo: probar muchas configuraciones y quedarse con la mejor
produce ganadores falsos por azar (multiple testing). Aquí se fija una sola.
NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

H = 21          # horizonte de predicción (días)
REFIT = 21      # refit del modelo cada N días
MIN_TRAIN = 756 # ~3 años mínimos de entrenamiento antes de predecir
COST_BPS = 2.0  # coste por rotación (ida) en puntos básicos
N_TICKERS = 120
ANOS = 12


def _features(px: pd.DataFrame) -> dict:
    """Features causales por fecha x ticker (solo información pasada)."""
    logp = np.log(px)
    f = {}
    for w in (21, 63, 126, 252):
        f[f"mom{w}"] = logp.diff(w)
    ret1 = logp.diff(1)
    for w in (21, 63):
        f[f"vol{w}"] = ret1.rolling(w).std()
    f["dist_max252"] = px / px.rolling(252).max() - 1.0
    f["ret5"] = logp.diff(5)
    d = px.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return f


def _panel(sintetico=False):
    if sintetico:
        rng = np.random.default_rng(4)
        n, k = 1800, 40
        idx = pd.bdate_range("2016-01-01", periods=n)
        cols = [f"SYN{i}" for i in range(k)]
        r = rng.normal(0.0003, 0.015, (n, k))
        return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
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
                if len(s) > 800:
                    cierres[tk] = s
            except Exception:
                continue
    if len(cierres) < 30:
        return None
    return pd.DataFrame(cierres).dropna(how="all").ffill().dropna(axis=1)


def _boot(x, n_boot=2000, bloque=21, seed=11):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        m = float(np.mean(x)) if n else 0.0
        return m, [m, m], 1.0
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bloque)); base = np.arange(bloque)
    med = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n, size=nb)[:, None] + base[None, :]).ravel() % n
        med[i] = x[idx].mean()
    lo, hi = np.percentile(med, [5, 95]); m = float(np.mean(x))
    p = float(np.mean(med <= 0)) if m > 0 else float(np.mean(med >= 0))
    return m, [round(float(lo), 4), round(float(hi), 4)], round(min(1.0, 2 * p), 4)


def evaluar_ml(sintetico=False):
    from sklearn.ensemble import HistGradientBoostingRegressor

    px = _panel(sintetico)
    if px is None or px.shape[1] < 20 or len(px) < MIN_TRAIN + 200:
        return None
    feats = _features(px)
    nombres = list(feats.keys())
    fechas = px.index
    tickers = list(px.columns)
    n, k = len(fechas), len(tickers)

    fwd = px.shift(-H) / px - 1.0                       # retorno futuro a H
    fwd_rel = fwd.sub(fwd.mean(axis=1), axis=0)         # relativo a la sección cruzada

    F = np.stack([feats[c].values for c in nombres], axis=2)   # (n, k, p)
    Y = fwd_rel.values

    modelo = None
    ret_ml, ret_bh, fechas_op = [], [], []
    t = MIN_TRAIN
    while t < n - H:
        if modelo is None or (t - MIN_TRAIN) % REFIT == 0:
            # entrenamiento: solo datos cuyo objetivo ya se conocía en t
            lim = t - H
            Xtr = F[:lim].reshape(-1, len(nombres))
            ytr = Y[:lim].reshape(-1)
            m = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
            if m.sum() < 5000:
                t += REFIT; continue
            Xtr, ytr = Xtr[m], ytr[m]
            if len(ytr) > 300000:                        # muestreo para acotar tiempo
                sel = np.random.default_rng(0).choice(len(ytr), 300000, replace=False)
                Xtr, ytr = Xtr[sel], ytr[sel]
            modelo = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            modelo.fit(Xtr, ytr)

        Xt = F[t]
        ok = np.isfinite(Xt).all(axis=1)
        if ok.sum() >= 20:
            pred = np.full(k, np.nan)
            pred[ok] = modelo.predict(Xt[ok])
            umbral = np.nanquantile(pred, 0.80)
            sel = np.where(np.isfinite(pred) & (pred >= umbral))[0]
            if len(sel) >= 3 and t + H < n:
                r_sel = (px.values[t + H, sel] / px.values[t, sel] - 1.0)
                r_all = (px.values[t + H, ok] / px.values[t, ok] - 1.0)
                coste = 2 * COST_BPS / 10000.0            # entrada + salida
                ret_ml.append(float(np.nanmean(r_sel)) - coste)
                ret_bh.append(float(np.nanmean(r_all)))
                fechas_op.append(fechas[t].strftime("%Y-%m-%d"))
        t += H                                            # operaciones no solapadas

    if len(ret_ml) < 20:
        return None
    ml = np.array(ret_ml); bh = np.array(ret_bh)
    exceso = (ml - bh) * 100.0
    m_ex, ic_ex, p_ex = _boot(exceso, bloque=1)
    acierto = float(np.mean(ml > bh) * 100)
    cagr_ml = float((np.prod(1 + ml) ** (252 / (H * len(ml))) - 1) * 100)
    cagr_bh = float((np.prod(1 + bh) ** (252 / (H * len(bh))) - 1) * 100)

    eq_ml = np.cumprod(1 + ml); eq_bh = np.cumprod(1 + bh)
    curva = [{"fecha": f, "valor": round(float(v), 4)} for f, v in zip(fechas_op, eq_ml)]
    curva2 = [{"fecha": f, "valor": round(float(v), 4)} for f, v in zip(fechas_op, eq_bh)]

    return {
        "id": "ml_forward",
        "etiqueta": "Machine learning (walk-forward)",
        "tipo": f"Gradient boosting · {px.shape[1]} valores · {len(ml)} periodos no solapados · neto de costes",
        "modelo": "ml",
        "color": "#b48ad6",
        "headline": {"valor": round(m_ex, 2),
                     "etiqueta": "Exceso medio por periodo frente a comprar y mantener",
                     "sufijo": "%", "decimales": 2},
        "significancia": {"p_valor": p_ex, "ic90": ic_ex,
                          "etiqueta": "exceso sobre comprar y mantener (%/periodo)"},
        "cards": [
            {"k": "Rentab. anual (CAGR) ML", "v": f"{cagr_ml:.1f}%", "tono": ""},
            {"k": "Rentab. anual (CAGR) comprar y mantener", "v": f"{cagr_bh:.1f}%", "tono": ""},
            {"k": "Periodos que baten al índice", "v": f"{acierto:.1f}%", "tono": ""},
            {"k": "Periodos evaluados", "v": str(len(ml)), "tono": ""},
            {"k": "Coste aplicado", "v": f"{COST_BPS:.0f} pb por operación", "tono": ""},
        ],
        "diagnostico": {},
        "curva": curva,
        "curva2": {"nombre": "Comprar y mantener el universo", "datos": curva2},
        "curva_color": "#b48ad6",
        "curva_unidad": "×",
        "curva_base": 1.0,
        "curva_titulo": "Capital acumulado: ML walk-forward vs comprar y mantener",
        "curva_sub": ("Protocolo fijado de antemano: una sola configuración, entrenamiento solo con el "
                      "pasado, refit periódico, operaciones no solapadas y costes aplicados. Si la línea "
                      "de color no supera a la gris, el ML no aporta sobre comprar y mantener."),
        "nota": ("Machine learning honesto: walk-forward estricto, sin lookahead y con una única "
                 "configuración prefijada (probar muchas y quedarse con la mejor fabrica ganadores "
                 "falsos). El veredicto se acepta tal cual. No es recomendación de inversión."),
    }
