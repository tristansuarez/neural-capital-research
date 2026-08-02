"""
Orquestador del laboratorio.

Corre cada experimento (un modelo sobre sus activos) con el motor de validacion
y vuelca todo a resultados.json, que es lo que leera la web.

Uso:
    python run_lab.py              # datos reales (Yahoo/Stooq)
    python run_lab.py --sintetico  # datos sinteticos, para probar sin red
"""

from __future__ import annotations
import sys
import json
import datetime as dt

import config
import data
import koncorde_forward
import garch_forward
import horizonte
import figuras
import implied_vol
import sentimiento
import ml_forward
import hipotesis
import pares_sectores
import hipotesis2
import momentum_tsm
import macro_rotacion
import combinaciones
from validation import evaluate
from models import BuyAndHold, GoldSilverPairs, PairsModel



def _limpiar_json(o):
    """NaN/Infinity son válidos para Python pero NO son JSON estándar: el navegador
    rechaza el archivo entero y la web se queda en blanco. Se sustituyen por None."""
    import math
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _limpiar_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar_json(v) for v in o]
    return o

def construir_experimentos():
    """Define el menu del desplegable: que modelo va con que activos."""
    return [
        {
            "id": "par_oro_plata",
            "etiqueta": "Oro-Plata (par cointegrado)",
            "activos": ["oro", "plata"],
            "modelo": GoldSilverPairs(),
            "tipo": "Reversion a la media / arbitraje estadistico",
        },
        {
            "id": "par_platino_paladio",
            "etiqueta": "Platino-Paladio (par cointegrado)",
            "activos": ["platino", "paladio"],
            "modelo": PairsModel("platino", "paladio"),
            "tipo": "Reversion a la media / arbitraje estadistico",
        },
        {
            "id": "oro_bh",
            "etiqueta": "Oro (benchmark comprar y mantener)",
            "activos": ["oro"],
            "modelo": BuyAndHold("oro"),
            "tipo": "Benchmark",
        },
        {
            "id": "plata_bh",
            "etiqueta": "Plata (benchmark comprar y mantener)",
            "activos": ["plata"],
            "modelo": BuyAndHold("plata"),
            "tipo": "Benchmark",
        },
    ]


def main(sintetico: bool = False):
    experimentos = construir_experimentos()
    salida = []

    for exp in experimentos:
        print(f"-> {exp['etiqueta']} ...", flush=True)
        if sintetico:
            syn = data.cargar_sinteticos()
            cols = exp["activos"]
            if set(cols) <= set(syn.columns):
                panel = syn[cols]
            else:
                # par sintético genérico: reutiliza el par cointegrado y renombra
                panel = syn[["oro", "plata"]].rename(
                    columns={"oro": cols[0], "plata": cols[1] if len(cols) > 1 else cols[0]})
        else:
            panel = data.cargar_panel(exp["activos"])

        informe = evaluate(
            panel, exp["modelo"],
            train_window=config.TRAIN_WINDOW,
            refit_every=config.REFIT_EVERY,
            cost_bps=config.COST_BPS,
        )
        informe["id"] = exp["id"]
        informe["etiqueta"] = exp["etiqueta"]
        informe["tipo"] = exp["tipo"]
        if exp["id"] == "par_oro_plata":
            ops, cols = koncorde_forward.operaciones_plata()
            informe["operaciones"] = ops
            informe["op_cols"] = cols
            hz = horizonte.horizonte_par(panel, "oro", "plata", "par oro-plata")
            if hz:
                informe["horizonte"] = hz
        if exp["id"] == "par_platino_paladio":
            hz = horizonte.horizonte_par(panel, "platino", "paladio", "par platino-paladio")
            if hz:
                informe["horizonte"] = hz
        if exp["id"] in ("oro_bh", "plata_bh"):
            informe["horizonte_na"] = (
                "En un comprar-y-mantener no hay señal ni entrada condicional: se está siempre "
                "invertido. El «horizonte» aquí sería solo cuánto tiempo aguantas, y más tiempo = "
                "más retorno acumulado (beta del mercado), sin nada condicional que medir.")
        salida.append(informe)
        h = informe["headline"]; s = informe["significancia"]
        print(f"   {h['etiqueta'][:22]} = {h['valor']}  p={s['p_valor']}  ic90={s['ic90']}", flush=True)

    # KONCORDE entra como un experimento mas (su forward-test en vivo).
    print("-> KONCORDE (S&P 500) ...", flush=True)
    kon = koncorde_forward.evaluar_koncorde(sintetico=sintetico)
    hzk = horizonte.horizonte_koncorde(sintetico=sintetico)
    if hzk:
        kon["horizonte"] = hzk
    salida.append(kon)
    if kon.get("sin_datos"):
        print("   (aun sin operaciones cerradas suficientes)", flush=True)
    else:
        print(f"   exceso medio = {kon['headline']['valor']}%  p={kon['significancia']['p_valor']}", flush=True)

    # GARCH de volatilidad por metal: no opera; mide si prevé la volatilidad mejor que lo ingenuo.
    resultados_garch = {}
    for metal in garch_forward.METALES:
        print(f"-> GARCH (volatilidad de {metal}) ...", flush=True)
        try:
            gar = garch_forward.evaluar_garch(sintetico=sintetico, activo=metal)
        except Exception as e:
            print(f"   fallo: {e}", flush=True)
            continue
        resultados_garch[metal] = gar
        salida.append(gar)
        if gar.get("sin_datos"):
            print("   (sin datos suficientes)", flush=True)
        else:
            print(f"   mejora vs ingenuo = {gar['headline']['valor']}%  "
                  f"p={gar['significancia']['p_valor']}", flush=True)

    # Panel conjunto de volatilidad de metales (comparativa).
    print("-> Panel de volatilidad de metales ...", flush=True)
    panel = garch_forward.panel_metales(resultados_garch)
    if panel:
        salida.append(panel)
        print(f"   {len(panel['metales'])} metales en el panel", flush=True)
    else:
        print("   (no hay suficientes metales con datos)", flush=True)

    # Volatilidad implícita: ¿el GARCH bate al mercado (VIX/GVZ), no solo al ingenuo?
    print("-> Volatilidad implícita (GARCH vs VIX/GVZ) ...", flush=True)
    try:
        _vis = implied_vol.evaluar_todas(sintetico=sintetico)
    except Exception as e:
        _vis = []; print(f"   (error vol implícita: {e})", flush=True)
    for vi in _vis:
        salida.append(vi)
        print(f"   {vi['id']}: mejora {vi['headline']['valor']} pts (p={vi['significancia']['p_valor']})", flush=True)

    # Machine learning con walk-forward estricto (protocolo fijado de antemano).
    print("-> Machine learning (walk-forward) ...", flush=True)
    try:
        mlr = ml_forward.evaluar_ml(sintetico=sintetico)
    except Exception as e:
        mlr = None
        print(f"   (error: {e})", flush=True)
    if mlr:
        salida.append(mlr)
        print(f"   exceso {mlr['headline']['valor']}%/periodo (p={mlr['significancia']['p_valor']})", flush=True)
    else:
        print("   (sin datos suficientes)", flush=True)

    # Hipótesis clásicas (anomalías con fundamento) con FDR conjunto.
    print("-> Hipótesis clásicas (deriva, reversión, estacionalidad) ...", flush=True)
    try:
        hip = hipotesis.backtest_hipotesis(sintetico=sintetico)
    except Exception as e:
        hip = None; print(f"   (error: {e})", flush=True)
    if hip:
        salida.append(hip)
        print(f"   {hip['n_celdas']} celdas, {hip['n_fdr']} sobreviven al FDR conjunto", flush=True)
    else:
        print("   (sin datos suficientes)", flush=True)

    # Pares sectoriales cointegrados (ETFs líquidos): reversión con FDR y costes.
    print("-> Pares sectoriales (cointegración entre sectores) ...", flush=True)
    try:
        ps = pares_sectores.evaluar_pares_sectores(sintetico=sintetico)
    except Exception as e:
        ps = None; print(f"   (error: {e})", flush=True)
    if ps:
        salida.append(ps)
        try:
            print(f"   {ps.get('tipo','')} | {ps.get('n_fdr','sin')} celdas superan el FDR", flush=True)
        except Exception:
            pass
    else:
        print("   (sin datos suficientes)", flush=True)

    # Hipótesis nuevas (momentum de índice, baja volatilidad, reversión mensual, tamaño).
    print("-> Hipótesis nuevas (momentum, baja vol, reversión, tamaño) ...", flush=True)
    try:
        hip2 = hipotesis2.backtest_hipotesis2(sintetico=sintetico)
    except Exception as e:
        hip2 = None; print(f"   (error: {e})", flush=True)
    if hip2:
        salida.append(hip2)
        try:
            print(f"   {hip2.get('n_celdas')} hipótesis, {hip2.get('n_fdr')} sobreviven al FDR", flush=True)
        except Exception:
            pass
    else:
        print("   (sin datos suficientes)", flush=True)

    # Momentum de serie temporal: la única hipótesis que cumplió lo que prometía.
    print("-> Momentum de serie temporal (protección) ...", flush=True)
    try:
        tsm = momentum_tsm.evaluar_tsm(sintetico=sintetico)
    except Exception as e:
        tsm = None; print(f"   (error: {e})", flush=True)
    if tsm:
        salida.append(tsm)
        try:
            print(f"   ahorro medio en caída: {tsm['headline']['valor']} pts", flush=True)
        except Exception:
            pass
    else:
        print("   (sin datos suficientes)", flush=True)

    # Rotación y filtros macro: estacionalidad, curva, crédito, dólar, defensivos.
    print("-> Rotación y filtros macro ...", flush=True)
    try:
        mac = macro_rotacion.evaluar_macro(sintetico=sintetico)
    except Exception as e:
        mac = None; print(f"   (error: {e})", flush=True)
    if mac:
        salida.append(mac)
        try:
            print(f"   {mac.get('n_celdas')} estrategias, {mac.get('n_fdr')} superan el FDR", flush=True)
        except Exception:
            pass
    else:
        print("   (sin datos suficientes)", flush=True)

    # Combinaciones y estacionalidad larga: exprimir lo que ha aportado.
    print("-> Combinaciones y estacionalidad larga ...", flush=True)
    try:
        cmb = combinaciones.evaluar_combinaciones(sintetico=sintetico)
    except Exception as e:
        cmb = None; print(f"   (error: {e})", flush=True)
    if cmb:
        salida.append(cmb)
        try:
            print(f"   {cmb.get('n_celdas')} variantes, {cmb.get('n_fdr')} superan el FDR", flush=True)
        except Exception:
            pass
    else:
        print("   (sin datos suficientes)", flush=True)

    # Sentimiento extremo (VIX contrario): hipótesis con fundamento, event study + FDR.
    print("-> Sentimiento extremo (VIX contrario) ...", flush=True)
    try:
        sent = sentimiento.backtest_sentimiento(sintetico=sintetico)
    except Exception as e:
        sent = None; print(f"   (error sentimiento: {e})", flush=True)
    if sent:
        salida.append(sent)
        print(f"   {sent['n_celdas']} celdas, {sent['n_fdr']} sobreviven al FDR", flush=True)
    else:
        print("   (sin datos de VIX suficientes)", flush=True)

    # Figuras técnicas: detección de chartismo + backtest honesto (event study).
    # graf_d / graf_i / graf_m recogen, de paso, las velas + geometría para el visor.
    graf_d, graf_i, graf_m = {}, {}, {}
    print("-> Figuras técnicas (S&P 500) ...", flush=True)
    try:
        fig = figuras.backtest_figuras(sintetico=sintetico, graf=graf_d)
    except Exception as e:
        fig = None; print(f"   (error figuras: {e})", flush=True)
    if fig:
        salida.append(fig)
        print(f"   {fig['n_celdas']} celdas evaluadas, {fig['n_fdr']} sobreviven al FDR", flush=True)
    else:
        print("   (sin datos suficientes)", flush=True)

    # Figuras intradía (velas de 1h, ~2 años): mismo motor, otra temporalidad.
    print("-> Figuras técnicas intradía (1h) ...", flush=True)
    try:
        figi = figuras.backtest_figuras(sintetico=sintetico, intradia=True, graf=graf_i)
    except Exception as e:
        figi = None; print(f"   (error figuras: {e})", flush=True)
    if figi:
        salida.append(figi)
        print(f"   {figi['n_celdas']} celdas evaluadas, {figi['n_fdr']} sobreviven al FDR", flush=True)
    else:
        print("   (sin datos intradía suficientes)", flush=True)

    # Figuras mensuales (velas de 1 mes, ~15 años): pocas y lentas, pero medidas.
    print("-> Figuras técnicas mensual ...", flush=True)
    try:
        figm = figuras.backtest_figuras(sintetico=sintetico, mensual=True, graf=graf_m)
    except Exception as e:
        figm = None; print(f"   (error figuras: {e})", flush=True)
    if figm:
        salida.append(figm)
        print(f"   {figm['n_celdas']} celdas evaluadas, {figm['n_fdr']} sobreviven al FDR", flush=True)
    else:
        print("   (sin datos mensuales suficientes)", flush=True)

    # Visor de gráficos: velas + figuras dibujables, por ticker y temporalidad.
    graf_doc = {
        "generado": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "diario": graf_d,
        "intradia": graf_i,
        "mensual": graf_m,
    }
    with open("graficos.json", "w", encoding="utf-8") as f:
        json.dump(_limpiar_json(graf_doc), f, ensure_ascii=False, allow_nan=False)
    print(f"   graficos.json: {len(graf_d)} diario / {len(graf_i)} intradía / {len(graf_m)} mensual", flush=True)

    doc = {
        "generado": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sintetico": bool(sintetico),
        "aviso": ((("DATOS SINTETICOS DE VERIFICACION (no son oro/plata reales). "
                    if sintetico else "")
                   + "Laboratorio de experimentacion de modelos. NO es "
                   "recomendacion de inversion. Todas las metricas son fuera de "
                   "muestra (walk-forward) y netas de costes. Un Sharpe con "
                   "p-valor alto NO es distinguible del azar.")),
        "experimentos": salida,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(_limpiar_json(doc), f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"\nresultados.json escrito con {len(salida)} experimentos.")


if __name__ == "__main__":
    main(sintetico="--sintetico" in sys.argv)
