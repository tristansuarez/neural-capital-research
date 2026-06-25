"""
Configuracion central del laboratorio.

Aqui se declara que activos hay y con que tickers se descargan. Anadir un
activo nuevo es tocar solo este archivo (mas su modelo, si lo lleva).
"""

# Tickers en Yahoo Finance (fuente primaria) y Stooq (respaldo).
# GC=F = futuro del oro, SI=F = futuro de la plata, ^GSPC = S&P 500.
ACTIVOS = {
    "oro":   {"yahoo": "GC=F",  "stooq": "xauusd"},
    "plata": {"yahoo": "SI=F",  "stooq": "xagusd"},
    "sp500": {"yahoo": "^GSPC", "stooq": "^spx"},
}

# Anos de historico a descargar.
ANOS_HISTORICO = 15

# Parametros del walk-forward (en dias de mercado).
TRAIN_WINDOW = 504    # ~2 anos de entrenamiento inicial
REFIT_EVERY = 21      # re-entrenar cada mes aprox.
COST_BPS = 2.0        # coste por rotacion completa, en puntos basicos

# Carpeta de cache de datos.
CACHE_DIR = "data_cache"
