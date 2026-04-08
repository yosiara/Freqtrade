# Archivo: user_data/strategies/GridStrategy.py

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta

class GridStrategy(IStrategy):
    """
    Estrategia de Grid adaptativo para mercados laterales
    """
    timeframe = '15m'
    
    # Take profit ajustado (más realista)
    minimal_roi = {
        "0": 0.015,   # 1.5% → salir
        "60": 0.01,   # 1% después de 1 hora
        "120": 0.005  # 0.5% después de 2 horas
    }
    
    stoploss = -0.015  # Stop loss -1.5%
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bandas de Bollinger (para detectar sobrecompra/sobreventa)
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Compra cuando precio toca banda inferior de Bollinger Y RSI < 40
        dataframe.loc[
            (
                (dataframe['close'] <= dataframe['bb_lowerband']) &
                (dataframe['rsi'] < 40) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Venta cuando precio toca banda superior O RSI > 70
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['bb_upperband']) |
                (dataframe['rsi'] > 70)
            ),
            'exit_long'] = 1
        return dataframe