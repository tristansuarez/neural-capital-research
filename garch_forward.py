"""
garch_forward.py
================
Modelo GARCH(1,1) de volatilidad, como experimento del laboratorio.

A diferencia de los demas, NO predice direccion ni opera: predice la VOLATILIDAD
(el tamano de los movimientos) un dia por delante. La direccion del precio es casi
impredecible, pero la volatilidad SI tiene estructura: se agrupa (dias movidos con
movidos, tranquilos con tranquilos). Eso es lo que captura un GARCH (Nobel de Engle).

Como se mide, con honestidad y fuera de muestra:
  - Walk-forward: se ajusta el GARCH sobre el pasado, se preve la varianza de manana,
    se avanza en el tiempo y se reajusta cada trimestre. Entre reajustes, los parametros
    quedan fijos y la varianza se actualiza cada dia con el retorno realizado (filtro).
  - Se compara contra un baseline INGENUO: la volatilidad de los ultimos 21 dias.
  - La metrica de error es la QLIKE (estandar para previsiones de volatilidad).
  - El titular es cuanto mejora el GARCH el error del ingenuo (%), con su significancia
    por bootstrap por bloques. Si el intervalo no cruza el cero, la mejora es real.

NO es asesoramiento financiero.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
import data

ANNUAL = float(np.sqrt(252))
NAIVE_WIN = 21          # ventana del baseline ingenuo (vol. reciente)
REFIT_GARCH = 63        # reajustar el GARCH cada ~trimestre (estable, mas rapido)
ACTIVO = "oro"


def _retornos(sintetico=False, activo=ACTIVO):
    serie = (data.cargar_sinteticos() if sintetico else data.cargar_panel([activo]))[activo]
    r = (np.log(serie / serie.shift(1)).dropna()) * 100.0   # retornos diarios en %
    return r


def _qlike(realized_var, forecast_var):
    """Perdida QLIKE: penaliza igual de bien sobre/infra-estimar; robusta para volatilidad."""
    fv = np.maximum(forecast_var, 1e-8)
    rv = np.maximum(realized_var, 1e-12)
    return rv / fv - np.log(rv / fv) - 1.0


def _bootstrap_mejora(perd_naive, perd_garch, n_boot=2000, seed=11, bloque=21):
    """Bootstrap por bloques de la mejora % = (error_ingenuo - error_garch)/error_ingenuo*100."""
    rng = np.random.default_rng(seed)
    n = len(perd_garch)
    if n < bloque + 1:
        return {"p_valor": 1.0, "ic90": [0.0, 0.0]}
    nb = max(1, n // bloque)
    mejoras = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - bloque + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + bloque) for s in starts])
        mn = perd_naive[idx].mean()
        mejoras.append(((mn - perd_garch[idx].mean()) / mn * 100.0) if mn != 0 else 0.0)
    mejoras = np.array(mejoras)
    lo, hi = np.percentile(mejoras, [5, 95])
    return {"p_valor": round(float(np.mean(mejoras <= 0)), 4),
            "ic90": [round(float(lo), 2), round(float(hi), 2)]}


def _sin_datos(msg):
    return {"id": "garch_vol", "etiqueta": "GARCH (volatilidad del oro)",
            "tipo": "Volatilidad · GARCH(1,1) · fuera de muestra",
            "modelo": "garch_1_1", "sin_datos": True, "sin_datos_txt": msg}


def evaluar_garch(sintetico=False):
    from arch import arch_model
    r = _retornos(sintetico=sintetico)
    fechas, vals, n = r.index, r.values, len(r)
    train = config.TRAIN_WINDOW
    if n < train + 100:
        return _sin_datos("No hay suficiente histórico para evaluar el GARCH todavía.")

    fc_g, fc_n, rv_real, ev_idx = [], [], [], []
    omega = alpha = beta = mu = sigma2 = last_r = None
    alpha_beta = None
    t = train
    while t < n:
        if sigma2 is None or (t - train) % REFIT_GARCH == 0:
            try:
                res = arch_model(vals[:t], mean="Constant", vol="Garch",
                                 p=1, q=1, rescale=False).fit(disp="off", show_warning=False)
                mu = float(res.params.get("mu", 0.0))
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                alpha_beta = alpha + beta
                sigma2 = float(res.conditional_volatility[-1] ** 2)
                last_r = vals[t - 1]
            except Exception:
                if sigma2 is None:
                    t += 1
                    continue
        # previsión de varianza para el día t (1 día por delante)
        f_var = omega + alpha * (last_r - mu) ** 2 + beta * sigma2
        win = vals[t - NAIVE_WIN:t]
        n_var = float(np.var(win)) if len(win) else f_var       # baseline ingenuo
        rv = float((vals[t] - mu) ** 2)                          # varianza realizada del día
        fc_g.append(f_var); fc_n.append(n_var); rv_real.append(rv); ev_idx.append(t)
        # actualizar el filtro para mañana
        sigma2 = f_var
        last_r = vals[t]
        t += 1

    if len(fc_g) < 100:
        return _sin_datos("El ajuste del GARCH no produjo suficientes previsiones.")

    fc_g = np.array(fc_g); fc_n = np.array(fc_n); rv_real = np.array(rv_real)
    perd_g, perd_n = _qlike(rv_real, fc_g), _qlike(rv_real, fc_n)
    mejora = (perd_n.mean() - perd_g.mean()) / perd_n.mean() * 100.0
    sig = _bootstrap_mejora(perd_n, perd_g)

    # curvas: vol prevista (GARCH) vs vol realizada suavizada (21d), anualizadas en %
    vol_prev = np.sqrt(fc_g) * ANNUAL
    rolled = pd.Series(vals).rolling(NAIVE_WIN).std().values
    vol_real = rolled[ev_idx] * ANNUAL
    f_ev = fechas[ev_idx]

    # alinear y limpiar NaN
    mask = np.isfinite(vol_prev) & np.isfinite(vol_real)
    vp, vr = vol_prev[mask], vol_real[mask]
    fechas_ev = [f_ev[i].strftime("%Y-%m-%d") for i in range(len(mask)) if mask[i]]
    corr = float(np.corrcoef(vp, vr)[0, 1]) if len(vp) > 2 else 0.0

    # acierto de régimen: ¿acierta cuándo la vol estará por encima/debajo de su mediana?
    med = np.median(vr)
    acierto = float(np.mean((vp > np.median(vp)) == (vr > med))) * 100.0

    PASO = max(1, len(vp) // 600)   # no más de ~600 puntos en el gráfico
    curva = [{"fecha": fechas_ev[i], "valor": round(float(vp[i]), 1)} for i in range(0, len(vp), PASO)]
    curva2 = [{"fecha": fechas_ev[i], "valor": round(float(vr[i]), 1)} for i in range(0, len(vr), PASO)]

    return {
        "id": "garch_vol",
        "etiqueta": "GARCH (volatilidad del oro)",
        "tipo": "Volatilidad · GARCH(1,1) · fuera de muestra",
        "modelo": "garch_1_1",
        "headline": {"valor": round(mejora, 1),
                     "etiqueta": "Mejora del error de previsión vs modelo ingenuo",
                     "sufijo": "%", "decimales": 1},
        "significancia": {"p_valor": sig["p_valor"], "ic90": sig["ic90"],
                          "etiqueta": "mejora vs ingenuo (%)"},
        "cards": [
            {"k": "Correlación previsión-realidad", "v": f"{corr:.2f}", "tono": ""},
            {"k": "Persistencia (α+β)", "v": f"{alpha_beta:.2f}", "tono": ""},
            {"k": "Acierto de régimen", "v": f"{acierto:.0f}%", "tono": ""},
            {"k": "Volatilidad media anual", "v": f"{np.nanmean(vr):.1f}%", "tono": ""},
            {"k": "Días evaluados", "v": str(len(vp)), "tono": ""},
        ],
        "diagnostico": {},
        "curva": curva,
        "curva2": {"nombre": "Volatilidad real (21d)", "datos": curva2},
        "curva_unidad": "%",
        "curva_base": round(float(np.nanmean(vr)), 1),
        "curva_titulo": "Volatilidad del oro: prevista vs realizada",
        "curva_sub": ("La línea dorada es la volatilidad que el modelo anticipa para cada día; "
                      "la azul, la que de verdad ocurrió. Cuanto más se siguen, mejor anticipa "
                      "las tormentas. La línea de puntos es la volatilidad media."),
    }
