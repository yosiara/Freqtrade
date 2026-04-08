# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

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

class EmaCruceEstrategia(IStrategy):
    """
    Estrategia de cruce de EMA con filtro RSI.
    - Compra: EMA rápida cruza arriba EMA lenta Y RSI > 50
    - Venta: EMA rápida cruza abajo EMA lenta
    - Stop loss personalizado basado en ATR
    - Tamaño de posición dinámico según volatilidad (ATR)
    """

    # --- Timeframe ---
    timeframe = '5m'

    # --- ROI estático (por ahora) ---
    minimal_roi = {
        "0": 0.10,    # 10% beneficio -> cerrar
        "30": 0.05,   # tras 30 min, 5%
        "60": 0.02,   # tras 1h, 2%
        "120": 0.01,  # tras 2h, 1%
        "240": 0      # tras 4h, salir aunque sea break even
    }

    # --- Stop loss (valor por defecto, será sobrescrito por custom_stoploss) ---
    stoploss = -0.05  # -5% máximo inicial

    # --- Parámetros optimizables (Hyperopt) ---
    buy_ema_fast = IntParameter(5, 30, default=12, space='buy')
    buy_ema_slow = IntParameter(20, 100, default=26, space='buy')
    buy_rsi_threshold = IntParameter(40, 70, default=50, space='buy')
    sell_ema_fast = IntParameter(5, 30, default=12, space='sell')
    sell_ema_slow = IntParameter(20, 100, default=26, space='sell')
    atr_mult_stoploss = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space='sell')
    risk_per_trade = DecimalParameter(0.01, 0.03, default=0.02, decimals=3, space='sell')  # 2% riesgo por trade

    # --- Protecciones (opcional) ---
    # position_adjustment_enable = False
    # use_exit_signal = False
    # exit_profit_only = False
    # ignore_roi_if_entry_signal = False

    use_custom_stoploss = False
    trailing_stop = True
    trailing_stop_positive = 0.01        # 1% de trailing una vez en ganancias
    trailing_stop_positive_offset = 0.02 # Se activa cuando el profit supera 2%
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Medias móviles exponenciales (EMA)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.buy_ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.buy_ema_slow.value)
        dataframe['ema55'] = ta.EMA(dataframe, timeperiod=55)

        # EMA para venta (pueden ser iguales o distintas)
        dataframe['ema_fast_sell'] = ta.EMA(dataframe, timeperiod=self.sell_ema_fast.value)
        dataframe['ema_slow_sell'] = ta.EMA(dataframe, timeperiod=self.sell_ema_slow.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ATR (para stop loss dinámico y tamaño de posición)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # ADX para medir la fuerza de la tendencia
        dataframe['adx'] = ta.ADX(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condiciones = (
            (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow']))
            & (dataframe['rsi'] > self.buy_rsi_threshold.value)
            & (dataframe['volume'] > 0)
            & (dataframe['adx'] > 20)
            & (dataframe['close'] > dataframe['ema55'])
        )
        dataframe.loc[condiciones, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condiciones = (
            qtpylib.crossed_below(dataframe['ema_fast_sell'], dataframe['ema_slow_sell'])
            & (dataframe['volume'] > 0)
            & (dataframe['adx'] < 25)
        )
        dataframe.loc[condiciones, 'exit_long'] = 1
        return dataframe

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     """
    #     Sube el stop loss al mínimo de cada vela.
    #     """

    #     # 2. Obtenemos el último mínimo alcanzado (es un valor absoluto)
    #     lowest_since_entry = trade.min_rate

    #     # 3. Calculamos a qué precio nos gustaría poner el stop ahora (un poco por debajo del mínimo)
    #     #    Por ejemplo, a un 0.2% por debajo de ese mínimo.
    #     stop_candidate_price = lowest_since_entry * (1 - 0.002)

    #     # 4. Comprobamos si este nuevo stop es mejor (más alto) que el que teníamos
    #     if stop_candidate_price > trade.stop_loss:
    #         # Si es mejor, calculamos la distancia porcentual respecto al precio actual
    #         new_stop_percent = (stop_candidate_price - current_rate) / current_rate
    #         # La función debe devolver un número negativo
    #         return max(new_stop_percent, self.stoploss)
    #     else:
    #         # Si no es mejor, mantenemos el stop actual.
    #         # Calculamos el porcentaje del stop actual respecto al precio actual.
    #         current_stop_percent = (trade.stop_loss - current_rate) / current_rate
    #         return max(current_stop_percent, self.stoploss)

    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     """
    #     Stop loss dinámico basado en ATR.
    #     El stop se sitúa a (atr_mult_stoploss * ATR) por debajo del precio de entrada.
    #     Se actualiza en cada barra.
    #     """
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = dataframe.iloc[-1].squeeze()
    #     atr = last_candle['atr']
    #     # El stop loss se expresa como porcentaje negativo respecto al precio actual
    #     stoploss_pct = - (self.atr_mult_stoploss.value * atr / current_rate)
    #     # Limitar stop loss máximo a -10% por seguridad
    #     return max(stoploss_pct, -0.10)

    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: float, max_stake: float,
    #                         entry_tag: str, side: str, **kwargs) -> float:
    #     """
    #     Tamaño de posición dinámico según volatilidad (ATR).
    #     Se arriesga un porcentaje fijo de la cartera (risk_per_trade) basado en la distancia del stop.
    #     """
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = dataframe.iloc[-1].squeeze()
    #     atr = last_candle['atr']
    #     # Distancia estimada del stop (en precio)
    #     stop_distance = self.atr_mult_stoploss.value * atr
    #     # Precio de entrada simulado = current_rate (asumimos entrada a mercado)
    #     # Riesgo monetario = risk_per_trade * wallet_balance
    #     wallet_balance = self.wallets.get_total_stake_amount()
    #     risk_amount = wallet_balance * self.risk_per_trade.value
    #     # Tamaño de posición = riesgo / (stop_distance / current_rate)
    #     stake = risk_amount / (stop_distance / current_rate)
    #     # Limitar entre min_stake y max_stake definidos en config.json
    #     return max(min_stake, min(stake, max_stake))