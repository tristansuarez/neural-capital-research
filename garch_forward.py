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

# Metales sobre los que corre el GARCH (oro ya estaba; los demas se anaden aqui).
METALES = ["oro", "plata", "platino", "paladio", "cobre"]
NOMBRE_METAL = {"oro": "oro", "plata": "plata", "platino": "platino",
                "paladio": "paladio", "cobre": "cobre"}
ARTICULO = {"oro": "del oro", "plata": "de la plata", "platino": "del platino",
            "paladio": "del paladio", "cobre": "del cobre"}
COLOR_METAL = {"oro": "#e8b23a", "plata": "#c0c5cc", "platino": "#8fb8c9",
               "paladio": "#b48ad6", "cobre": "#d08a5a"}

HORIZONTES_G = [1, 5, 10, 21, 63]
LABEL_H = {1: "1 día", 5: "1 sem", 10: "2 sem", 21: "1 mes", 63: "3 meses"}

# Estructura de plazos de la PREVISIÓN actual (hasta el máximo plazo útil ~1 año).
PREV_H = [1, 5, 10, 21, 42, 63, 126, 252]
LABEL_PREV = {1: "1 día", 5: "1 sem", 10: "2 sem", 21: "1 mes",
              42: "2 meses", 63: "3 meses", 126: "6 meses", 252: "1 año"}
SKILL_SET = set(HORIZONTES_G)


def _serie_sintetica(activo):
    """Serie de precios sintética con clustering de volatilidad (para pruebas offline)."""
    seed = abs(hash(activo)) % (2**31)
    rng = np.random.default_rng(seed)
    n = 3500
    sigma2 = 1.0
    om, al, be = 0.02, 0.08, 0.90
    r = np.empty(n)
    for i in range(n):
        sigma2 = om + al * (r[i - 1] ** 2 if i else 0.0) + be * sigma2
        r[i] = rng.normal(0, np.sqrt(sigma2))
    precio = 100.0 * np.exp(np.cumsum(r) / 100.0)
    idx = pd.bdate_range(end="2024-07-01", periods=n)
    return pd.Series(precio, index=idx)


def _retornos(sintetico=False, activo=ACTIVO):
    if sintetico:
        if activo in ("oro", "plata"):
            serie = data.cargar_sinteticos()[activo]
        else:
            serie = _serie_sintetica(activo)
    else:
        serie = data.cargar_panel([activo])[activo]
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


def _sin_datos(msg, activo="oro"):
    return {"id": f"garch_{activo}", "etiqueta": f"GARCH (volatilidad {ARTICULO[activo]})",
            "tipo": "Volatilidad · GARCH(1,1) · fuera de muestra",
            "modelo": "garch_1_1", "sin_datos": True, "sin_datos_txt": msg}


def evaluar_garch(sintetico=False, activo="oro"):
    from arch import arch_model
    try:
        r = _retornos(sintetico=sintetico, activo=activo)
    except Exception:
        return _sin_datos(f"No se pudieron cargar datos {ARTICULO[activo]} todavía.", activo)
    fechas, vals, n = r.index, r.values, len(r)
    train = config.TRAIN_WINDOW
    if n < train + 100:
        return _sin_datos("No hay suficiente histórico para evaluar el GARCH todavía.", activo)

    fc_g, fc_n, rv_real, ev_idx = [], [], [], []
    HZ_g = {h: [] for h in HORIZONTES_G}   # QLIKE del GARCH por horizonte
    HZ_n = {h: [] for h in HORIZONTES_G}   # QLIKE del ingenuo por horizonte
    RATIO = {h: [] for h in PREV_H}        # vol_real/vol_prevista por horizonte (calibra bandas)
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
        # multi-horizonte: previsión de la varianza MEDIA de los próximos h días
        ab = alpha + beta
        lr = omega / (1.0 - ab) if 0 < ab < 1 else f_var
        for hh in PREV_H:
            if t + hh > n:
                continue
            geom = (1.0 - ab ** hh) / (1.0 - ab) if 0 < ab < 1 else float(hh)
            fc_avg = lr + (f_var - lr) * (geom / hh)               # varianza media prevista
            real_avg = float(np.mean((vals[t:t + hh] - mu) ** 2))  # varianza media realizada
            if fc_avg > 0:
                RATIO[hh].append(np.sqrt(max(real_avg, 0.0)) / np.sqrt(fc_avg))
            if hh in SKILL_SET:
                naive_avg = float(np.var(vals[t - NAIVE_WIN:t]))   # vol reciente (21d): mismo ingenuo a todo plazo
                HZ_g[hh].append(_qlike(real_avg, fc_avg))
                HZ_n[hh].append(_qlike(real_avg, naive_avg))
        # actualizar el filtro para mañana
        sigma2 = f_var
        last_r = vals[t]
        t += 1

    if len(fc_g) < 100:
        return _sin_datos("El ajuste del GARCH no produjo suficientes previsiones.", activo)

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

    nombre = NOMBRE_METAL[activo]
    color = COLOR_METAL[activo]
    return {
        "id": f"garch_{activo}",
        "etiqueta": f"GARCH (volatilidad {ARTICULO[activo]})",
        "tipo": "Volatilidad · GARCH(1,1) · fuera de muestra",
        "modelo": "garch_1_1",
        "metal": activo,
        "color": color,
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
        "curva_color": color,
        "curva_unidad": "%",
        "curva_base": round(float(np.nanmean(vr)), 1),
        "curva_titulo": f"Volatilidad {ARTICULO[activo]}: prevista vs realizada",
        "curva_sub": (f"La línea de color es la volatilidad que el modelo anticipa para cada día; "
                      f"la azul, la que de verdad ocurrió. Cuanto más se siguen, mejor anticipa "
                      f"las tormentas. La línea de puntos es la volatilidad media."),
        "horizonte": _horizonte_garch(HZ_g, HZ_n, alpha_beta),
        "prevision": _prevision_actual(vals, fechas, RATIO, activo),
    }


def _prevision_actual(vals, fechas, RATIO, activo="oro"):
    """Estructura de plazos de la volatilidad esperada DESDE HOY (1 día..1 año), con banda."""
    from arch import arch_model
    try:
        res = arch_model(vals, mean="Constant", vol="Garch", p=1, q=1,
                         rescale=False).fit(disp="off", show_warning=False)
        mu = float(res.params.get("mu", 0.0))
        omega = float(res.params["omega"])
        ab = float(res.params["alpha[1]"]) + float(res.params["beta[1]"])
        f_next = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    except Exception:
        return None
    lr = omega / (1.0 - ab) if 0 < ab < 1 else f_next      # varianza de largo plazo
    puntos = []
    for hh in PREV_H:
        geom = (1.0 - ab ** hh) / (1.0 - ab) if 0 < ab < 1 else float(hh)
        avg_var = lr + (f_next - lr) * (geom / hh)
        vol = float(np.sqrt(max(avg_var, 1e-12)) * ANNUAL)
        rr = np.array(RATIO.get(hh, []))
        lo_r, hi_r = (np.percentile(rr, [5, 95]) if len(rr) > 30 else (0.7, 1.4))
        puntos.append({"h": hh, "etiqueta": LABEL_PREV[hh], "vol": round(vol, 1),
                       "lo": round(float(vol * lo_r), 1), "hi": round(float(vol * hi_r), 1)})
    vol_lr = float(np.sqrt(max(lr, 1e-12)) * ANNUAL)
    vol_1d = puntos[0]["vol"]
    if vol_1d > vol_lr:
        regimen = (f"Movido: la volatilidad está por encima de su media (~{vol_lr:.1f}%). "
                   f"El modelo espera que se vaya calmando hacia ese nivel.")
    else:
        regimen = (f"Tranquilo: la volatilidad está por debajo de su media (~{vol_lr:.1f}%). "
                   f"El modelo espera que repunte hacia ese nivel.")
    return {
        "titulo": f"Previsión actual de volatilidad {ARTICULO[activo]}",
        "color": COLOR_METAL[activo],
        "sub": ("Volatilidad anualizada que el GARCH espera DESDE LA ÚLTIMA SESIÓN, para cada "
                "horizonte. Es una estimación viva: cambia cada día con los datos nuevos."),
        "fecha": fechas[-1].strftime("%Y-%m-%d"),
        "actual": vol_1d,
        "largo_plazo": round(vol_lr, 1),
        "regimen": regimen,
        "unidad": "%",
        "puntos": puntos,
        "nota": ("Las bandas reflejan cuánto se ha desviado históricamente la volatilidad real de la "
                 "prevista a cada plazo (percentiles 5–95). A partir de ~1 año la previsión es ya, en "
                 "esencia, la volatilidad media de largo plazo: el GARCH no aporta más allá."),
    }


def _horizonte_garch(HZ_g, HZ_n, alpha_beta):
    """Mejora del error de previsión vs el ingenuo, por horizonte (1d..3 meses)."""
    puntos = []
    for hh in HORIZONTES_G:
        qg, qn = np.array(HZ_g[hh]), np.array(HZ_n[hh])
        if len(qg) < 50 or qn.mean() == 0:
            continue
        mej = (qn.mean() - qg.mean()) / qn.mean() * 100.0
        b = _bootstrap_mejora(qn, qg, bloque=max(21, hh))
        puntos.append({"h": hh, "etiqueta": LABEL_H[hh], "valor": round(float(mej), 1),
                       "ic_lo": b["ic90"][0], "ic_hi": b["ic90"][1], "n": int(len(qg))})
    if not puntos:
        return None
    crece = puntos[-1]["valor"] > puntos[0]["valor"]
    tendencia = "crece" if crece else "se reduce"
    return {
        "titulo": "Habilidad del GARCH según el horizonte",
        "sub": ("Mejora del error de previsión frente al modelo ingenuo (la volatilidad reciente), "
                "para la volatilidad MEDIA de los próximos N días. Si la barra no toca el cero, el "
                "GARCH aún aporta a ese plazo."),
        "unidad": "%",
        "puntos": puntos,
        "nota": (f"El baseline ingenuo usa la volatilidad reciente y se queda plano; el GARCH, en "
                 f"cambio, revierte hacia la volatilidad media de largo plazo. Por eso su ventaja "
                 f"relativa {tendencia} con el plazo (persistencia {alpha_beta:.2f}): la curva "
                 f"muestra hasta dónde aporta el modelo."),
    }


def panel_metales(resultados):
    """Construye el experimento 'panel' conjunto a partir de los GARCH individuales.

    `resultados` es {activo: dict_resultado}. Reutiliza la previsión (estructura de
    plazos) y la curva de volatilidad realizada de cada metal: no recalcula nada.
    """
    metales = []
    for activo in METALES:
        ex = resultados.get(activo)
        if not ex or ex.get("sin_datos"):
            continue
        prev = ex.get("prevision") or {}
        pts = prev.get("puntos", [])
        # serie histórica (submuestreada a ~220 puntos para que el conjunto no pese)
        hist = ex.get("curva2", {}).get("datos", [])
        paso = max(1, len(hist) // 220)
        hist_ds = hist[::paso]
        metales.append({
            "nombre": NOMBRE_METAL[activo].capitalize(),
            "metal": activo,
            "color": COLOR_METAL[activo],
            "actual": prev.get("actual"),
            "largo_plazo": prev.get("largo_plazo"),
            "prev": [{"etiqueta": p["etiqueta"], "vol": p["vol"]} for p in pts],
            "hist": hist_ds,
        })
    if len(metales) < 2:
        return None
    # etiquetas de plazo comunes (del primero que tenga previsión)
    etiquetas = next((["", *[p["etiqueta"] for p in m["prev"]]][1:] for m in metales if m["prev"]), [])
    return {
        "id": "panel_metales",
        "etiqueta": "Panel de volatilidad de metales",
        "tipo": "Volatilidad · GARCH(1,1) · comparativa de metales",
        "modelo": "garch_1_1",
        "panel": True,
        "metales": metales,
        "plazos": etiquetas,
        "prev_titulo": "Previsión actual de volatilidad por metal",
        "prev_sub": ("Estructura de plazos que el GARCH espera hoy para cada metal (1 día → 1 año). "
                     "La pendiente dice el régimen: si baja, el metal está movido y se espera calma; "
                     "si sube, está tranquilo y se espera repunte."),
        "hist_titulo": "Volatilidad realizada por metal (anualizada, 21 días)",
        "hist_sub": ("La volatilidad que de verdad ocurrió en cada metal a lo largo del tiempo. "
                     "Permite comparar quién es estructuralmente más nervioso."),
        "unidad": "%",
        "nota": ("Industriales (cobre, paladio) suelen ser más volátiles que los refugios (oro). "
                 "El paladio, por su mercado pequeño y concentrado, es de los más extremos. "
                 "No es recomendación de inversión."),
    }
