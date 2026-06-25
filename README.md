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

- `index.html` — la web (un solo archivo): portada animada + panel con
  desplegable, veredicto de significancia y curvas. Lee `resultados.json`.
- `config.py` — activos, tickers y parámetros del walk-forward.
- `data.py` — descarga de precios (Yahoo → Stooq → caché) + generador sintético.
- `models/` — cada modelo cuantitativo expone la misma interfaz (`fit`/`weights`).
  - `naive.py` — comprar y mantener (benchmark).
  - `cointegration.py` — par oro-plata: cointegración + reversión a la media.
- `validation.py` — walk-forward, métricas y significancia por bootstrap.
- `koncorde_forward.py` — adapta el forward-test del KONCORDE al mismo esquema.
- `escaner_senales_telegram.py` — el bot: escanea el S&P 500, registra señales en
  `senales_log.csv` y las publica en Telegram.
- `run_lab.py` — corre todos los experimentos (oro/plata + KONCORDE) y escribe
  `resultados.json`.

## Workflows (GitHub Actions)

- `lab.yml` (06:00 UTC) — corre `run_lab.py` y publica `resultados.json` para la web.
- `koncorde.yml` (00:00 UTC) — corre el escáner, publica en Telegram y guarda
  `senales_log.csv`. Usa los secretos `TELEGRAM_TOKEN` y `TELEGRAM_CHANNEL`.

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
