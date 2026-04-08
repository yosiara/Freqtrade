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

class TrendMomentumStrategy(IStrategy):
    """
    Estrategia de Tendencia y Momento con confirmación de reversa.
    Timeframe principal: 15m
    Timeframe superior (tendencia): 1h
    """
    
    # ==================== CONFIGURACIÓN BÁSICA ====================
    timeframe = '15m'
    informative_timeframe = '1h'
    
    # ==================== GESTIÓN DE RIESGO ====================
    stoploss = -0.02
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    
    # ==================== PARÁMETROS OPTIMIZABLES ====================
    buy_ema_trend = IntParameter(40, 60, default=55, space='buy')
    buy_rsi = IntParameter(25, 40, default=35, space='buy')
    buy_volume_factor = DecimalParameter(1.0, 2.0, default=1.2, space='buy', decimals=1)
    sell_rsi = IntParameter(65, 80, default=75, space='sell')
    
    # ==================== ROI (Take Profit) ====================
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.01,
        "120": 0
    }
    
    # ==================== VELAS NECESARIAS ====================
    startup_candle_count: int = 200
    
    # ==================== INDICADORES PRINCIPALES ====================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula indicadores en timeframe principal (15m) y timeframe superior (1h)
        """
        
        # ===== INDICADORES EN TIMEFRAME PRINCIPAL (15m) =====
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_55'] = ta.EMA(dataframe, timeperiod=55)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Patrones de velas
        dataframe['bullish_engulfing'] = ta.CDLENGULFING(dataframe)
        dataframe['hammer'] = ta.CDLHAMMER(dataframe)
        dataframe['shooting_star'] = ta.CDLSHOOTINGSTAR(dataframe)
        
        # ===== INDICADORES EN TIMEFRAME SUPERIOR (1h) - MANUAL =====
        # Obtenemos el dataframe del timeframe 1h para el mismo par
        informative_df = self.dp.get_pair_dataframe(
            pair=metadata['pair'], 
            timeframe=self.informative_timeframe
        )
        
        if not informative_df.empty:
            # Calculamos indicadores en el dataframe de 1h
            informative_df['ema_55'] = ta.EMA(informative_df, timeperiod=55)
            informative_df['ema_200'] = ta.EMA(informative_df, timeperiod=200)
            informative_df['rsi'] = ta.RSI(informative_df, timeperiod=14)
            
            # Renombramos columnas para que tengan sufijo '_1h'
            informative_df = informative_df.add_suffix('_1h')
            
            # Hacemos merge con el dataframe principal usando la columna 'date'
            # Resampleamos para alinear los timestamps (cada 1h se repite en las velas de 15m)
            dataframe = dataframe.merge(
                informative_df[['date_1h', 'ema_55_1h', 'ema_200_1h', 'rsi_1h']],
                left_on='date',
                right_on='date_1h',
                how='left'
            )
            
            # Forward fill para propagar el valor de la hora actual a todas las velas de 15m dentro de esa hora
            dataframe['ema_55_1h'] = dataframe['ema_55_1h'].fillna(method='ffill')
            dataframe['ema_200_1h'] = dataframe['ema_200_1h'].fillna(method='ffill')
            dataframe['rsi_1h'] = dataframe['rsi_1h'].fillna(method='ffill')
            
            # Eliminamos la columna temporal auxiliar
            dataframe.drop('date_1h', axis=1, inplace=True)
        
        return dataframe
    
    # ==================== CONDICIONES DE COMPRA ====================
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones de COMPRA (señal = 1)
        """
        
        # Verificar que las columnas del timeframe 1h existen
        if 'ema_55_1h' not in dataframe.columns:
            # Si no hay datos del 1h, no compramos
            dataframe['buy'] = 0
            return dataframe
        
        # Condiciones de TENDENCIA GENERAL (filtro desde 1H)
        tendencia_alcista = (
            (dataframe['close'] > dataframe['ema_55_1h']) &
            (dataframe['ema_55_1h'] > dataframe['ema_200_1h'])
        )
        
        # Condiciones de PULLBACK en 15m
        pullback = (
            (dataframe['close'] > dataframe['ema_20']) &
            (dataframe['close'] < dataframe['ema_55']) &
            (dataframe['low'].rolling(5).min() <= dataframe['ema_55'])
        )
        
        # Condiciones de MOMENTUM y VOLUMEN
        momentum = (
            (dataframe['rsi'] > self.buy_rsi.value) &
            (dataframe['volume'] > dataframe['volume_mean'] * self.buy_volume_factor.value)
        )
        
        # Condiciones de PATRÓN DE VELA
        patron_vela = (
            (dataframe['bullish_engulfing'] == 100) |
            (dataframe['hammer'] == 100)
        )
        
        # SEÑAL FINAL
        condiciones = (
            tendencia_alcista &
            pullback &
            momentum &
            patron_vela
        )
        
        dataframe.loc[condiciones, 'buy'] = 1
        return dataframe
    
    # ==================== CONDICIONES DE VENTA ====================
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones de VENTA (señal = 1)
        """
        
        condiciones = (
            (dataframe['rsi'] > self.sell_rsi.value) |
            (dataframe['close'] < dataframe['ema_20']) |
            (dataframe['shooting_star'] == 100)
        )
        
        # Si existe la columna del 1h, añadimos condición de debilidad de tendencia
        if 'ema_55_1h' in dataframe.columns:
            condiciones = condiciones | (dataframe['close'] < dataframe['ema_55_1h'])
        
        dataframe.loc[condiciones, 'sell'] = 1
        return dataframe
    
    # ==================== STOP LOSS DINÁMICO ====================
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stop loss basado en ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle.get('atr', 0)
        
        if atr == 0:
            return self.stoploss
        
        sl = (2 * atr / current_rate)
        sl = max(0.01, min(0.05, sl))
        return sl