# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib

class PriceAction(IStrategy):
    timeframe = "15m"

    # can_short = True

    minimal_roi = {  # tiempo - beneficio -> cerrar
        "0": 0.04,   # máximo, 4%
        "120": 0.03, # tras 2h, 3%
        "180": 0.02, # tras 3h, 2%
        "240": 0,    # tras 4h, salir aunque sea break even
    }

    stoploss = -0.02                         # -2% máximo inicial
    trailing_stop = True
    trailing_stop_positive = 0.01            # Mover SL luego de 1%
    trailing_stop_positive_offset = 0.02     # Luego de 2% usar SL de -1%
    trailing_only_offset_is_reached = False  # No mover SL luego de 1%, esperar a 2%
    # use_custom_stoploss = False

    def fibonacci(self, last_swing_high_val, last_swing_low_val,
                  lvl = 0.618, high = False, low = False):
        """Cálculo de niveles Fibo en el pullback"""
        diff = last_swing_high_val - last_swing_low_val
        price = last_swing_high_val - (diff * lvl)
        return high > price if high else low > price

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Configuración para 15min: 7 velas de ventana (3 izq, centro, 3 der)
        window = 7
        half_window = window // 2  # Esto da 3

        # Umbral de separación para swings del mismo tipo (0.5%)
        separation_pct = 0.005

        # 1. Detectar mínimos y máximos significativos)
        # 1.1. Swing Lows: La vela actual es más baja que las 3 de antes y las 3 de después
        dataframe["swing_low"] = (
            dataframe["low"] == dataframe["low"].rolling(window=window, center=True).min()
        )
        # 1.2. Swing High: La vela actual es más alta que las 3 de antes y las 3 de después
        dataframe["swing_high"] = (
            dataframe["high"] == dataframe["high"].rolling(window=window, center=True).max()
        )

        # 2. Extraer y propagar los precios de los pivotes encontrados
        # 2.1. Precio del último mínimo
        dataframe["last_swing_low_val"] = (
            dataframe["low"].where(dataframe["swing_low"]).ffill()
        )
        # 2.2. Precio del último máximo
        dataframe["last_swing_high_val"] = (
            dataframe["high"].where(dataframe["swing_high"]).ffill()
        )

        # 3. Crear la lógica de Tendencia Alcista
        # 3.1. Detección de nuevo pivote Higher High (HH) con separación para evitar ruido
        dataframe["higher_high"] = (
            dataframe["last_swing_high_val"] > dataframe["last_swing_high_val"].shift(1) * (1 + separation_pct)
        ).where(dataframe["swing_high"]).ffill() # Propagamos para confirmar "salud de tendencia"

        # 3.2. Detección de nuevo pivote Higher Low (HL) con separación para evitar ruido
        higher_low = (
            dataframe["last_swing_low_val"] > dataframe["last_swing_low_val"].shift(1) * (1 + separation_pct)
        )

        # 3.3. Comprar en pullback. Nuevo mínimo creciente(HL) luego de un máximo creciente(HH)
        dataframe["buy_pullback"] = (
            dataframe["swing_low"] & dataframe["higher_high"] & higher_low
        ).shift(half_window) # Mandamos la señal al futuro

        # 3.4. Tomamos como SL el precio del swing low
        dataframe["buy_swing_low_val"] = (
            dataframe["last_swing_low_val"].where(dataframe["buy_pullback"])
        ).ffill() # Propagamos hacia adelante para señal de salida en caso de ruptura

        # 4. Crear la lógica de Tendencia Bajista
        # 4.1. Detección de nuevo pivote Lower Low (LL) con separación para evitar ruido
        dataframe["lower_low"] = (
            dataframe["last_swing_low_val"] < dataframe["last_swing_low_val"].shift(1) * (1 - separation_pct)
        ).where(dataframe["swing_low"]).ffill() # Propagamos para confirmar "salud de tendencia"

        # 4.2. Detección de nuevo pivote Lower High (LH) con separación para evitar ruido
        lower_high = (
            dataframe["last_swing_high_val"] < dataframe["last_swing_high_val"].shift(1) * (1 - separation_pct)
        )

        # 4.3. Vender en pullback. Nuevo máximo decreciente(LH) luego de un mínimo decreciente(LL)
        dataframe["shell_pullback"] = (
            dataframe["swing_high"] & dataframe["lower_low"] & lower_high
        ).shift(half_window)

        # 4.4. Tomamos como SL el precio del swing high
        dataframe["shell_swing_high_val"] = (
            dataframe["last_swing_high_val"].where(dataframe["shell_pullback"])
        ).ffill() # Propagamos hacia adelante para señal de salida en caso de ruptura

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["buy_pullback"])                 # Confirmación en pullback
                & (dataframe["close"] > dataframe["open"])  # Vela verde
                & (dataframe["volume"] > 0)                 # Solo operar con volumen
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["shell_pullback"])               # Confirmación en pullback
                & (dataframe["close"] < dataframe["open"])  # Vela roja
                & (dataframe["volume"] > 0)                 # Solo operar con volumen
            ),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["close"] < dataframe["buy_swing_low_val"]     # Rompimiento del Swing Low
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                dataframe["close"] > dataframe["shell_swing_high_val"]  # Rompimiento del Swing High
            ),
            "exit_short",
        ] = 1
        return dataframe
