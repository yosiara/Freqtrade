# custom_indicators.py
import numpy as np
import pandas as pd
import talib.abstract as ta


def pivot_sr_volume(
    dataframe: pd.DataFrame,
    pivot_length: int = 50,
    vol_len: int = 2,
    box_width: float = 1.0
) -> pd.DataFrame:
    """
    Calcula pivotes, niveles fantasma y soportes/resistencias con filtro de volumen.

    Parámetros
    ----------
    dataframe : pd.DataFrame
        DataFrame con columnas OHLCV.
    pivot_length : int
        Longitud de la ventana para detectar pivotes (LuxAlgo).
    vol_len : int
        Período para calcular los umbrales de volumen delta (ChartPrime).
    box_width : float
        Factor multiplicador del ATR para definir la anchura de la zona S/R.

    Retorna
    -------
    pd.DataFrame
        DataFrame original con las siguientes columnas añadidas:

        LuxAlgo (Pivotes y ghost level)
        -------------------------------
        - pivot_high   : Precio del pivote alto confirmado.
        - pivot_low    : Precio del pivote bajo confirmado.
        - pivot_os     : Estado del último pivote (1=alto, 0=bajo, -1=inicial).
        - missed_high  : Precio donde se detectó un "missed pivot high".
        - missed_low   : Precio donde se detectó un "missed pivot low".
        - ghost_level  : Nivel fantasma horizontal actual (precio).

        ChartPrime (S/R con volumen)
        ----------------------------
        - sr_sup        : Nivel de soporte activo (basado en pivote bajo + volumen).
        - sr_res        : Nivel de resistencia activo (basado en pivote alto + volumen).
        - breakout_res  : Ruptura alcista de resistencia.
        - res_holds     : Resistencia rechaza el precio.
        - sup_holds     : Soporte sostiene el precio.
        - breakout_sup  : Ruptura bajista de soporte.
        - res_is_sup    : Resistencia previa actúa como soporte (cambio de rol).
        - sup_is_res    : Soporte previo actúa como resistencia (cambio de rol).
    """
    df = dataframe.copy()

    # ---------------------------------------------------------------------------
    # 1. Volumen delta
    # ---------------------------------------------------------------------------
    # Volumen en velas alcistas menos volumen en velas bajistas
    df['delta_vol'] = np.where(
        df['close'] > df['open'], df['volume'],
        np.where(df['close'] < df['open'], -df['volume'], 0)
    )

    # Umbrales móviles
    df['vol_hi'] = df['delta_vol'].rolling(vol_len).max()
    df['vol_lo'] = df['delta_vol'].rolling(vol_len).min()

    # ATR para el ancho de las zonas S/R
    df['atr'] = ta.ATR(df, timeperiod=200)
    df['width'] = df['atr'] * box_width

    # ---------------------------------------------------------------------------
    # 2. Pivotes
    # ---------------------------------------------------------------------------
    def unique_pivots(series, length):
        half = length
        roll = series.rolling(window=2*half+1, center=True)
        max_vals = roll.max()
        min_vals = roll.min()
        is_max = (series == max_vals)
        is_min = (series == min_vals)
        unique_max = roll.apply(lambda x: np.sum(x == x.max()) == 1, raw=True)
        unique_min = roll.apply(lambda x: np.sum(x == x.min()) == 1, raw=True)
        return is_max & unique_max, is_min & unique_min

    # Pivotes primarios
    ph_big, pl_big = unique_pivots(df['high'], pivot_length)
    df['ph_big'] = ph_big
    df['pl_big'] = pl_big

    # Estado
    df['pivot_os'] = -1
    df.loc[df['ph_big'], 'pivot_os'] = 1
    df.loc[df['pl_big'], 'pivot_os'] = 0
    df['pivot_os'] = df['pivot_os'].replace(-1, np.nan).ffill().fillna(0).astype(int)

    df['pivot_high'] = np.where(df['ph_big'], df['high'], np.nan)
    df['pivot_low']  = np.where(df['pl_big'], df['low'], np.nan)

    # Pivotes secundarios (para missed points)
    ph_small, pl_small = unique_pivots(df['high'], pivot_length // 2)
    df['ph_small'] = ph_small
    df['pl_small'] = pl_small

    # ---------------------------------------------------------------------------
    # 3. Segmentación
    # ---------------------------------------------------------------------------
    df['pivot_big_event'] = df['ph_big'] | df['pl_big']
    df['segment_id'] = df['pivot_big_event'].cumsum()

    # Desplazamos los extremos del segmento anterior a la barra del nuevo pivote grande
    df['prev_seg_max'] = df.groupby('segment_id')['high'].transform('max').shift(1)
    df['prev_seg_min'] = df.groupby('segment_id')['low'].transform('min').shift(1)

    # índice donde ocurrió ese máximo anterior para colocar allí el missed level
    df['prev_seg_max_idx'] = df.groupby('segment_id')['high'].transform('idxmax').shift(1)
    df['prev_seg_min_idx'] = df.groupby('segment_id')['low'].transform('idxmin').shift(1)

    # ---------------------------------------------------------------------------
    # 4. Missed Levels
    # ---------------------------------------------------------------------------
    cond_missed_high = df['pl_big'] & (df['prev_seg_max'] > df['low'])
    cond_missed_low  = df['ph_big'] & (df['prev_seg_min'] < df['high'])

    df['missed_high'] = np.nan
    df['missed_low']  = np.nan

    # Filtros de no coincidencia con pivotes
    if cond_missed_high.any():
        target_idx = df.loc[cond_missed_high, 'prev_seg_max_idx'].dropna().astype(int)
        target_vals = df.loc[cond_missed_high, 'prev_seg_max'].values
        is_pivot = df.loc[target_idx, 'ph_big'].values | df.loc[target_idx, 'pl_big'].values
        valid = ~is_pivot
        df.loc[target_idx[valid], 'missed_high'] = target_vals[valid]

    if cond_missed_low.any():
        target_idx = df.loc[cond_missed_low, 'prev_seg_min_idx'].dropna().astype(int)
        target_vals = df.loc[cond_missed_low, 'prev_seg_min'].values
        is_pivot = df.loc[target_idx, 'ph_big'].values | df.loc[target_idx, 'pl_big'].values
        valid = ~is_pivot
        df.loc[target_idx[valid], 'missed_low'] = target_vals[valid]

    # Ghost level
    last_missed = df['missed_high'].combine_first(df['missed_low'])
    df['ghost_level'] = last_missed.ffill()

    # ---------------------------------------------------------------------------
    # 5. ChartPrime S/R
    # ---------------------------------------------------------------------------
    cond_sup = df['pl_big'] & (df['delta_vol'] > df['vol_hi']) # Support levels with Positive Volume
    cond_res = df['ph_big'] & (df['delta_vol'] < df['vol_lo']) # Resistance levels with Negative Volume

    # Soporte y resistencia
    df['sr_sup'] = np.nan
    df['sr_res'] = np.nan
    df.loc[cond_sup, 'sr_sup'] = df['low']
    df.loc[cond_res, 'sr_res'] = df['high']

    df['sr_sup'] = df['sr_sup'].ffill()
    df['sr_res'] = df['sr_res'].ffill()

    # Niveles desplazados para simular caja según fuerza
    df['sup_level_1'] = df['sr_sup'] - df['width']
    df['res_level_1'] = df['sr_res'] + df['width']

    # Eventos de ruptura/rechazo
    df['breakout_res'] = (df['low'] > df['res_level_1']) & (df['low'].shift(1) <= df['res_level_1'].shift(1))
    df['res_holds'] = (df['high'] >= df['sr_res']) & (df['close'] < df['sr_res'])
    df['sup_holds'] = (df['low']  <= df['sr_sup']) & (df['close'] > df['sr_sup'])
    df['breakout_sup'] = (df['high'] < df['sup_level_1']) & (df['high'].shift(1) >= df['sup_level_1'].shift(1))

    # Cambio de rol
    df['res_is_sup'] = (df['breakout_res'].astype(int) - df['res_holds'].astype(int)).cumsum().clip(0, 1).astype(bool)
    df['sup_is_res'] = (df['breakout_sup'].astype(int) - df['sup_holds'].astype(int)).cumsum().clip(0, 1).astype(bool)

    # Limpieza
    cols_to_drop = [
        'delta_vol', 'vol_hi', 'vol_lo', 'atr', 'width',
        'ph_big', 'pl_big', 'ph_small', 'pl_small', 'pivot_big_event', 'segment_id',
        'prev_seg_max', 'prev_seg_min', 'prev_seg_max_idx', 'prev_seg_min_idx',
        'sup_level_1', 'res_level_1'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df

def fibonacci(last_swing_high_val, last_swing_low_val,
              lvl = 0.618, high = False, low = False):
    """Cálculo de niveles Fibo en el pullback"""
    diff = last_swing_high_val - last_swing_low_val
    price = last_swing_high_val - (diff * lvl)
    return high > price if high else low > price
