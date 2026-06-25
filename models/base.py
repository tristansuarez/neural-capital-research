"""
Interfaz común de todos los modelos del laboratorio.

La regla de oro es la honestidad metodológica: un modelo NUNCA puede mirar el
futuro. Por eso la interfaz se diseña para que el motor de validación
(validation.py) sea quien controla el tiempo. El modelo solo recibe el historial
disponible HASTA un instante y devuelve los pesos que tendria para el dia
SIGUIENTE. El motor aplica esos pesos al rendimiento del dia siguiente.

Cada modelo expone:
  - name, description        -> metadatos para la web
  - fit(train_df)            -> estima sus parametros con una ventana de
                                entrenamiento (se llama periodicamente en el
                                walk-forward)
  - weights(history_df)      -> dict {activo: peso} objetivo para el dia
                                siguiente, usando SOLO informacion de history_df
                                (cuya ultima fila es el ultimo dia conocido)

Un peso +1 = largo una unidad del activo; -1 = corto; 0 = fuera.
Para modelos de par (oro-plata) los pesos sobre las dos patas hacen la
estrategia neutral al mercado.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    name: str = "base"
    description: str = ""
    # Activos que necesita el modelo (columnas que espera en el DataFrame)
    assets: list[str] = []

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Estima parametros usando solo las filas de train_df."""
        raise NotImplementedError

    @abstractmethod
    def weights(self, history_df: pd.DataFrame) -> dict[str, float]:
        """
        Devuelve los pesos objetivo para el dia siguiente al ultimo de
        history_df. Solo puede usar informacion contenida en history_df.
        """
        raise NotImplementedError
