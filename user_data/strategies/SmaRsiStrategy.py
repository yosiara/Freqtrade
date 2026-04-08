from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SmaRsiStrategy(IStrategy):
    """
    Estrategia simple pero funcional:
    - Compra cuando SMA rápida cruza arriba de SMA lenta (tendencia alcista)
    - Filtro adicional: RSI > 50 (confirmación de momentum)
    - Vende cuando SMA rápida cruza debajo de SMA lenta
    """
    
    # Timeframe
    timeframe = '15m'
    
    # ROI (Take profit)
    minimal_roi = {
        "0": 0.03,   # 3% de ganancia, salir
        "30": 0.02,  # 2% después de 30 minutos
        "60": 0.01,  # 1% después de 60 minutos
        "120": 0.005 # 0.5% después de 120 minutos
    }
    
    # Stop loss
    stoploss = -0.02  # -2%
    
    # Parámetros optimizables
    buy_fast_ma = IntParameter(10, 30, default=15, space="buy")
    buy_slow_ma = IntParameter(30, 60, default=30, space="buy")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMA rápida y lenta
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=self.buy_fast_ma.value)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=self.buy_slow_ma.value)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Condiciones de compra
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['fast_ma'], dataframe['slow_ma']) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Condiciones de venta
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['fast_ma'], dataframe['slow_ma']) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe