# Laboratorio de modelos

Banco de pruebas honesto para experimentar con modelos de mercado. **No es
recomendación de inversión.** El objetivo no es "predecir el precio" —la
dirección diaria es esencialmente un paseo aleatorio— sino comprobar con rigor
si una hipótesis concreta tiene algo de señal o es ruido.

## La idea central

El rigor no está en los predictores, está en la maquinaria que los valida.
Cada modelo se evalúa **siempre fuera de muestra** (walk-forward), **neto de
costes**, y se le exige superar un **benchmark** y ser **estadísticamente
distinguible del azar**.

## Estructura

- `index.html` — la web (un solo archivo). Lee `resultados.json` y pinta el
  desplegable, el veredicto de significancia y la curva de capital.
- `config.py` — activos, tickers y parámetros del walk-forward.
- `data.py` — descarga de precios (Yahoo → Stooq → caché) + generador sintético.
- `models/` — cada modelo expone la misma interfaz (`fit` / `weights`).
  - `naive.py` — comprar y mantener (benchmark).
  - `cointegration.py` — par oro-plata: cointegración + reversión a la media.
- `validation.py` — walk-forward, métricas (Sharpe, drawdown…) y significancia
  por bootstrap de bloques.
- `run_lab.py` — corre todos los experimentos y escribe `resultados.json`.

## Cambiar el nombre

El nombre de la web está en una sola línea, en `index.html`:
`const NOMBRE = "CRISOL";`. Cámbialo por el que quieras.

## Cómo se lee un resultado

- **Sharpe**: rentabilidad ajustada al riesgo. Es la métrica central.
- **p-valor** (significancia): probabilidad de que ese Sharpe salga así por puro
  azar. Si es alto, **el modelo no vale aunque su Sharpe parezca bueno**.
- **Benchmark**: un modelo direccional debe batir al "comprar y mantener". Uno
  neutral al mercado (como el par) debe tener Sharpe distinguible de cero.

## Ejecutar

```bash
pip install -r requirements.txt
python run_lab.py              # datos reales
python run_lab.py --sintetico  # prueba del motor sin red
```

La GitHub Action lo corre solo cada día laborable y publica `resultados.json`.
