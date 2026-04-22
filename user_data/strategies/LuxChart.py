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

from custom_indicators import pivot_sr_volume


class LuxChart(IStrategy):
    timeframe = "15m"

    # can_short = True

    minimal_roi = {  # tiempo - beneficio -> cerrar
        "0": 0.05,   # máximo, 5%
        "180": 0.03, # tras 2h, 3%
        "240": 0.02, # tras 3h, 2%
        # "240": 0,    # tras 4h, salir aunque sea break even
    }

    stoploss = -0.02                         # -2% máximo inicial
    trailing_stop = True
    trailing_stop_positive = 0.01            # Mover SL luego de 1%
    trailing_stop_positive_offset = 0.02     # Luego de 2% usar SL de -1%
    trailing_only_offset_is_reached = True   # No mover SL luego de 1%, esperar a 2%
    # use_custom_stoploss = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = pivot_sr_volume(dataframe=dataframe, pivot_length=50, vol_len=2, box_width=1.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['breakout_res']           # Ruptura de resistencia
                | dataframe['sup_holds']            # Soporte confirmado
            ) &
            (
                (dataframe['volume'] > 0)           # Confirmar volumen siempre
                & ~dataframe['retest_failed_sup']   # Evitar entrar en soporte comprometido
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                dataframe['breakout_sup']           # Ruptura de soporte
                | dataframe['res_holds']            # Resistencia confirmada
            ) &
            (
                (dataframe['volume'] > 0)           # Confirmar volumen siempre
                & ~dataframe['retest_failed_res']   # Evitar entrar en resistencia comprometida
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['breakout_sup_active']    # Retest failed
            ),
        "exit_long"] = 1

        dataframe.loc[
            (
                dataframe['breakout_res_active']    # Retest failed
            ),
        "exit_short"] = 1

        return dataframe
