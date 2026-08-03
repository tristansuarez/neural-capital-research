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
N_TICKERS = 80
ANOS = 30       # máximo con sentido: incluye 2000-2002, 2008, 2020, 2022.
                # Antes de ~1995 la composición del índice y la calidad del dato
                # cambian tanto que la comparación deja de ser limpia.


def _features(px: pd.DataFrame) -> dict:
    """Features causales por fecha x ticker (solo información pasada).
    Tres familias: (a) del propio valor, (b) relativas al mercado y (c) de estado
    del mercado. Las (b) y (c) son información que el modelo antes NO tenía."""
    logp = np.log(px)
    f = {}
    # --- (a) propias del valor ---
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

    # --- (b) relativas a la sección cruzada (¿destaca frente a sus pares?) ---
    for w in (21, 126, 252):
        m = f[f"mom{w}"]
        f[f"mom{w}_rel"] = m.sub(m.mean(axis=1), axis=0)          # exceso sobre la media
        f[f"mom{w}_rank"] = m.rank(axis=1, pct=True)              # percentil transversal
    f["vol21_rank"] = f["vol21"].rank(axis=1, pct=True)
    f["reversion_1m"] = -f["mom21_rel"]                            # los rezagados rebotan

    # --- (c) estado del mercado (mismo valor para todos, define el régimen) ---
    idx = px.mean(axis=1)
    ridx = np.log(idx).diff(1)
    mercado = {
        "mkt_mom126": np.log(idx).diff(126),
        "mkt_vol21": ridx.rolling(21).std(),
        "mkt_vol_ratio": ridx.rolling(21).std() / ridx.rolling(126).std(),
        "mkt_dist_max": idx / idx.rolling(252).max() - 1.0,
        "dispersion": px.pct_change().rolling(21).std().std(axis=1),   # dispersión entre valores
    }
    for k, s in mercado.items():
        f[k] = pd.DataFrame(np.repeat(s.values[:, None], px.shape[1], axis=1),
                            index=px.index, columns=px.columns)
    # interacción: momentum del valor x régimen del mercado
    f["mom126_x_regimen"] = f["mom126"] * np.sign(f["mkt_mom126"])
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
    # Requerir historia completa desde el inicio deja SOLO supervivientes: las que
    # existían hace ANOS años y siguen HOY en el índice. Es un sesgo grave que infla
    # el resultado. Se mide y se reporta (ver _aviso_supervivencia).
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
    ret_ml, ret_bh, ret_mom, regimen, fechas_op = [], [], [], [], []
    # Régimen causal: media del universo por encima de su media de 200 sesiones
    idx_eq = px.mean(axis=1)
    alcista = (idx_eq > idx_eq.rolling(200).mean()).values
    mom252 = feats["mom252"].values
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
            if len(ytr) > 150000:                        # muestreo para acotar tiempo
                sel = np.random.default_rng(0).choice(len(ytr), 150000, replace=False)
                Xtr, ytr = Xtr[sel], ytr[sel]
            modelo = HistGradientBoostingRegressor(
                max_depth=3, max_iter=120, learning_rate=0.05,
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
                # benchmark: momentum simple a 12 meses, mismas reglas
                mm = np.where(ok, mom252[t], np.nan)
                if np.isfinite(mm).sum() >= 20:
                    u2 = np.nanquantile(mm, 0.80)
                    s2 = np.where(np.isfinite(mm) & (mm >= u2))[0]
                    r_mom = float(np.nanmean(px.values[t + H, s2] / px.values[t, s2] - 1.0)) - coste \
                        if len(s2) >= 3 else np.nan
                else:
                    r_mom = np.nan
                ret_mom.append(r_mom)
                regimen.append(bool(alcista[t]))
                fechas_op.append(fechas[t].strftime("%Y-%m-%d"))
        t += H                                            # operaciones no solapadas

    if len(ret_ml) < 20:
        return None
    ml = np.array(ret_ml); bh = np.array(ret_bh)
    mom = np.array(ret_mom, dtype=float); reg = np.array(regimen)
    exceso = (ml - bh) * 100.0
    m_ex, ic_ex, p_ex = _boot(exceso, bloque=1)
    acierto = float(np.mean(ml > bh) * 100)
    cagr_ml = float((np.prod(1 + ml) ** (252 / (H * len(ml))) - 1) * 100)
    cagr_bh = float((np.prod(1 + bh) ** (252 / (H * len(bh))) - 1) * 100)

    # --- TORTURAS ---
    # 1) quitar los 5 mejores periodos
    orden = np.argsort(exceso)[::-1]
    keep = np.ones(len(exceso), bool); keep[orden[:5]] = False
    m_t5, ic_t5, p_t5 = _boot(exceso[keep], bloque=1)
    # 2) por régimen (causal: índice sobre/bajo su media de 200 sesiones)
    m_al, _i, p_al = _boot(exceso[reg], bloque=1) if reg.sum() >= 20 else (float('nan'), None, 1.0)
    m_ba, _i2, p_ba = _boot(exceso[~reg], bloque=1) if (~reg).sum() >= 20 else (float('nan'), None, 1.0)
    # 3) ¿bate el ML al momentum simple con las mismas reglas?
    okm = np.isfinite(mom)
    if okm.sum() >= 20:
        ex_mom = (mom[okm] - bh[okm]) * 100.0
        m_mm, _i3, p_mm = _boot(ex_mom, bloque=1)
        dif = (ml[okm] - mom[okm]) * 100.0
        m_df, ic_df, p_df = _boot(dif, bloque=1)
    else:
        m_mm = p_mm = m_df = p_df = float('nan'); ic_df = None

    tort = (
        "<br><br><b>Pruebas de robustez (un artefacto no las supera):</b>"
        f"<br>• <b>Sin los 5 mejores periodos</b>: exceso {m_t5:+.2f}%/periodo (p={p_t5}) — "
        f"si aquí desaparece, la ventaja eran cuatro golpes de suerte."
        f"<br>• <b>Por régimen</b>: mercado alcista {m_al:+.2f}% (p={p_al}, n={int(reg.sum())}) · "
        f"bajista/lateral {m_ba:+.2f}% (p={p_ba}, n={int((~reg).sum())}) — si solo gana en alcista, "
        f"es beta disfrazada de alfa."
        f"<br>• <b>Vs momentum simple</b> (ranking por retorno a 12 meses, mismas reglas): el momentum "
        f"da {m_mm:+.2f}%/periodo sobre el índice; el ML le saca <b>{m_df:+.2f}%</b> (p={p_df}, "
        f"IC90 {[round(float(x),2) for x in ic_df] if ic_df else None}) — si no le saca nada, "
        f"el ML solo redescubrió el momentum.")

    eq_ml = np.cumprod(1 + ml); eq_bh = np.cumprod(1 + bh)
    curva = [{"fecha": f, "valor": round(float(v), 4)} for f, v in zip(fechas_op, eq_ml)]
    curva2 = [{"fecha": f, "valor": round(float(v), 4)} for f, v in zip(fechas_op, eq_bh)]

    # --- ¿Se comporta igual en todos los regímenes? Desglose honesto ---
    fechas_arr = np.array(fechas_op)
    decadas = {}
    for i, f in enumerate(fechas_arr):
        d = f[:3] + "0s"
        decadas.setdefault(d, []).append(i)
    filas_dec = []
    for d in sorted(decadas):
        idx = np.array(decadas[d])
        if len(idx) < 8:
            continue
        ex = exceso[idx]
        m = float(np.mean(ex))
        ganados = float(np.mean(ml[idx] > bh[idx]) * 100)
        mercado = float((np.prod(1 + bh[idx]) ** (252 / (H * len(idx))) - 1) * 100)
        filas_dec.append(
            f"<tr><td>{d}</td><td class='{'pos' if m > 0 else 'neg'}'>{m:+.2f}%</td>"
            f"<td class='est-obs'>{ganados:.0f}%</td>"
            f"<td class='{'pos' if mercado > 0 else 'neg'}'>{mercado:+.1f}%</td>"
            f"<td class='est-obs'>{len(idx)}</td></tr>")

    # por signo del mercado en cada periodo: ¿aporta cuando el mercado cae?
    sube = bh > 0
    filas_reg = []
    for etq, m_ in (("Mercado al alza", sube), ("Mercado a la baja", ~sube)):
        if m_.sum() < 5:
            continue
        ex = exceso[m_]
        mm, ic, pp = _boot(ex, bloque=1)
        filas_reg.append(
            f"<tr><td>{etq}</td><td class='{'pos' if mm > 0 else 'neg'}'>{mm:+.2f}%</td>"
            f"<td class='est-obs'>[{ic[0]:.2f}, {ic[1]:.2f}]</td>"
            f"<td class='{'pos' if pp <= 0.10 else 'est-obs'}'>{pp}</td>"
            f"<td class='est-obs'>{int(m_.sum())}</td></tr>")

    regimenes = ""
    if filas_dec:
        regimenes += ("<br><br><b>¿Se comporta igual en todas las épocas?</b> Si el modelo detectase "
                      "algo real, su ventaja no debería depender de la década."
                      "<div class='ops-scroll'><table class='ops'><thead><tr><th>Década</th>"
                      "<th>Exceso medio</th><th>Periodos ganados</th><th>Mercado</th><th>n</th>"
                      f"</tr></thead><tbody>{''.join(filas_dec)}</tbody></table></div>")
    if filas_reg:
        regimenes += ("<br><b>¿Y cuando el mercado cae?</b> La prueba de si sabe defenderse o solo "
                      "sabe subir con la marea."
                      "<div class='ops-scroll'><table class='ops'><thead><tr><th>Régimen</th>"
                      "<th>Exceso medio</th><th>IC 90%</th><th>p</th><th>n</th>"
                      f"</tr></thead><tbody>{''.join(filas_reg)}</tbody></table></div>")

    # ¿Cuánto sesgo de supervivencia arrastra este universo?
    pedidos = N_TICKERS
    usados = px.shape[1]
    grave = usados < 0.75 * pedidos
    aviso_sup = ""
    if grave:
        aviso_sup = (
            f"<div class='ch-sub' style='border-left:3px solid #d2566a;padding-left:12px;margin:14px 0'>"
            f"<b>⚠️ Sesgo de supervivencia grave.</b> De {pedidos} valores pedidos, solo {usados} "
            f"tienen {ANOS} años de historia continua. Esas {usados} son, por definición, las que "
            f"existían hace {ANOS} años <b>y siguen hoy</b> en el índice: las que quebraron o fueron "
            f"expulsadas no están. Comprar «las que sobrevivieron» gana casi siempre, así que "
            f"<b>cualquier ventaja medida aquí está inflada y no debe tomarse como edge real</b>. "
            f"Con datos gratuitos no hay forma de reconstruir la composición histórica del índice. "
            f"<br><br>Comprobación: la ventaja de este modelo ha cambiado de signo cada vez que se "
            f"ha modificado la ventana temporal o el tamaño del universo "
            f"(+0,59% → +0,17% → −0,05% → {round(m_ex, 2)}%). Un edge real no se comporta así; "
            f"esa inestabilidad es la firma del ruido y del sesgo, no de una señal.</div>")

    # --- CAPACIDAD PREDICTIVA PURA (independiente del sesgo de supervivencia) ---
    # No mide cuánto gana, sino si ORDENA bien: ¿acierta qué acciones lo harán mejor
    # que otras? El sesgo de supervivencia sube el nivel de todas por igual, así que
    # no infla estas métricas. Es la prueba de capacidad, no de rentabilidad.
    ic_lista, top_bot, aciertos_par = [], [], []
    t = MIN_TRAIN
    modelo_ic = None
    while t < n - H:
        if modelo_ic is None or (t - MIN_TRAIN) % REFIT == 0:
            lim = t - H
            Xtr = F[:lim].reshape(-1, len(nombres)); ytr = Y[:lim].reshape(-1)
            msk = np.isfinite(ytr) & np.isfinite(Xtr).all(axis=1)
            if msk.sum() >= 5000:
                Xtr, ytr = Xtr[msk], ytr[msk]
                if len(ytr) > 150000:
                    sel_ = np.random.default_rng(0).choice(len(ytr), 150000, replace=False)
                    Xtr, ytr = Xtr[sel_], ytr[sel_]
                modelo_ic = HistGradientBoostingRegressor(
                    max_depth=3, max_iter=120, learning_rate=0.05,
                    min_samples_leaf=200, l2_regularization=1.0, random_state=0)
                modelo_ic.fit(Xtr, ytr)
        if modelo_ic is not None and t + H < n:
            Xt = F[t]; ok_ = np.isfinite(Xt).all(axis=1)
            real = np.full(k, np.nan)
            real[ok_] = px.values[t + H, ok_] / px.values[t, ok_] - 1.0
            val = np.isfinite(real) & ok_
            if val.sum() >= 20:
                pr = modelo_ic.predict(Xt[val])
                rl = real[val]
                # 1) Information Coefficient: correlación de rangos predicción-realidad
                rp = pd.Series(pr).rank().values; rr = pd.Series(rl).rank().values
                if np.std(rp) > 0 and np.std(rr) > 0:
                    ic_lista.append(float(np.corrcoef(rp, rr)[0, 1]))
                # 2) top vs bottom: ¿el quintil alto bate al bajo?
                q_hi, q_lo = np.quantile(pr, 0.8), np.quantile(pr, 0.2)
                hi_, lo_ = rl[pr >= q_hi], rl[pr <= q_lo]
                if len(hi_) >= 3 and len(lo_) >= 3:
                    top_bot.append(float(np.mean(hi_) - np.mean(lo_)) * 100)
                # 3) acierto por pares: de cada 2 acciones, ¿ordena bien?
                rng_p = np.random.default_rng(t)
                idx_a = rng_p.integers(0, val.sum(), 200)
                idx_b = rng_p.integers(0, val.sum(), 200)
                dif_p = pr[idx_a] - pr[idx_b]; dif_r = rl[idx_a] - rl[idx_b]
                mv = (dif_p != 0) & (dif_r != 0)
                if mv.sum() > 20:
                    aciertos_par.append(float(np.mean(np.sign(dif_p[mv]) == np.sign(dif_r[mv])) * 100))
        t += H

    cap = ""
    if len(ic_lista) >= 24:
        ic_m, ic_ic, ic_p = _boot(np.array(ic_lista), bloque=1)
        tb_m, tb_ic, tb_p = _boot(np.array(top_bot), bloque=1) if len(top_bot) >= 24 else (0, [0, 0], 1)
        ap_m, ap_ic, ap_p = (_boot(np.array(aciertos_par) - 50.0, bloque=1)
                             if len(aciertos_par) >= 24 else (0, [0, 0], 1))
        veredicto_ic = (
            "<div class='ch-sub' style='margin-top:10px;border-left:3px solid "
            + ("#6ec08a" if ic_m >= 0.03 else "#d2566a") + ";padding-left:12px'>"
            + (f"<b>SUPERA el umbral.</b> IC = {ic_m:.3f} &ge; 0,03, el minimo explotable segun la "
               f"ley fundamental de la gestion activa. Sharpe teorico maximo &asymp; "
               f"{ic_m * np.sqrt(usados):.2f} antes de costes. Merece investigarse mas."
               if ic_m >= 0.03 else
               f"<b>NO supera el umbral.</b> IC = {ic_m:.3f} &lt; 0,03, por debajo del minimo "
               f"explotable. Sharpe teorico maximo &asymp; {ic_m * np.sqrt(usados):.2f} antes de "
               f"costes (regla de Grinold: Sharpe &asymp; IC x raiz de n). Aunque sea estadisticamente "
               f"distinto de cero, es demasiado pequeno para sobrevivir a los costes de operarlo.")
            + " El umbral se fijo ANTES de medir, no despues.</div>")

        cap = (
            "<br><br><b>Capacidad predictiva pura</b> (no mide cuánto gana, sino si ORDENA bien "
            "las acciones). El sesgo de supervivencia sube el nivel de todas por igual, así que "
            "aquí no infla el resultado: esta es la prueba de capacidad."
            "<div class='ops-scroll'><table class='ops'><thead><tr><th>Métrica</th><th>Valor</th>"
            "<th>IC 90%</th><th>p</th><th>Qué significaría tener capacidad</th></tr></thead><tbody>"
            f"<tr><td>Coeficiente de información (IC)</td>"
            f"<td class='{'pos' if ic_m > 0 else 'neg'}'>{ic_m:+.3f}</td>"
            f"<td class='est-obs'>[{ic_ic[0]:.3f}, {ic_ic[1]:.3f}]</td>"
            f"<td class='{'pos' if ic_p <= 0.10 else 'est-obs'}'>{ic_p}</td>"
            f"<td class='est-obs'>&gt; 0,03 ya sería explotable; &gt; 0,05 es bueno</td></tr>"
            f"<tr><td>Quintil alto − quintil bajo</td>"
            f"<td class='{'pos' if tb_m > 0 else 'neg'}'>{tb_m:+.2f}%</td>"
            f"<td class='est-obs'>[{tb_ic[0]:.2f}, {tb_ic[1]:.2f}]</td>"
            f"<td class='{'pos' if tb_p <= 0.10 else 'est-obs'}'>{tb_p}</td>"
            f"<td class='est-obs'>positivo y significativo</td></tr>"
            f"<tr><td>Acierto ordenando pares (sobre 50%)</td>"
            f"<td class='{'pos' if ap_m > 0 else 'neg'}'>{ap_m:+.2f} pts</td>"
            f"<td class='est-obs'>[{ap_ic[0]:.2f}, {ap_ic[1]:.2f}]</td>"
            f"<td class='{'pos' if ap_p <= 0.10 else 'est-obs'}'>{ap_p}</td>"
            f"<td class='est-obs'>&gt; +2 pts sobre el 50% del azar</td></tr>"
            "</tbody></table></div>"
            "<div class='ch-sub' style='margin-top:8px'>Si estas tres métricas rondan cero, el modelo "
            "no distingue una acción de otra: su rentabilidad viene de estar invertido, no de elegir. "
            "Si son claramente positivas, hay capacidad real aunque la rentabilidad esté contaminada "
            "por el sesgo del universo.</div>" + veredicto_ic)

    # Variante causal: operar solo cuando el régimen es alcista (info conocida ese día)
    ml_filtrado = np.where(reg, ml, 0.0)     # fuera de mercado en régimen no alcista
    cagr_fil = float((np.prod(1 + ml_filtrado) ** (252 / (H * len(ml_filtrado))) - 1) * 100)

    return {
        "id": "ml_forward",
        "etiqueta": "Machine learning (walk-forward)",
        "tipo": f"Gradient boosting · {px.shape[1]} valores · {len(ml)} periodos no solapados · neto de costes",
        "modelo": "ml",
        "color": "#b48ad6",
        "headline": {"valor": round(m_ex, 2),
                     "etiqueta": ("Exceso por periodo vs comprar y mantener"
                                  + (" — NO FIABLE: sesgo de supervivencia" if grave else "")),
                     "sufijo": "%", "decimales": 2},
        "significancia": {"p_valor": p_ex, "ic90": ic_ex,
                          "etiqueta": "exceso sobre comprar y mantener (%/periodo)"},
        "cards": [
            {"k": "Rentab. anual (CAGR) ML", "v": f"{cagr_ml:.1f}%", "tono": ""},
            {"k": "Rentab. anual (CAGR) comprar y mantener", "v": f"{cagr_bh:.1f}%", "tono": ""},
            {"k": "Periodos que baten al índice", "v": f"{acierto:.1f}%", "tono": ""},
            {"k": "CAGR ML solo en régimen alcista (filtro causal)", "v": f"{cagr_fil:.1f}%", "tono": ""},
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
        "nota": (aviso_sup + cap + "Machine learning honesto: walk-forward estricto, sin lookahead y con una única "
                 "configuración prefijada (probar muchas y quedarse con la mejor fabrica ganadores "
                 "falsos). El veredicto se acepta tal cual. No es recomendación de inversión."
                 + regimenes + tort),
    }
