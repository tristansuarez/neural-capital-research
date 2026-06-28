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
from validation import evaluate
from models import BuyAndHold, GoldSilverPairs, PairsModel


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
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"\nresultados.json escrito con {len(salida)} experimentos.")


if __name__ == "__main__":
    main(sintetico="--sintetico" in sys.argv)
