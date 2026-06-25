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
from validation import evaluate
from models import BuyAndHold, GoldSilverPairs


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
            panel = data.cargar_sinteticos()[exp["activos"]]
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
        salida.append(informe)
        m = informe["metricas"]
        s = informe["significancia"]
        print(f"   Sharpe={m['sharpe']}  CAGR={m['cagr']}  "
              f"maxDD={m['max_drawdown']}  p={s['p_valor']}", flush=True)

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
