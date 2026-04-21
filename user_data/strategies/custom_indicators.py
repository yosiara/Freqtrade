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
        - pivot_os     : Estado del último pivote (1=alto, 0=bajo).
        - missed_high  : Precio donde se detectó un "missed pivot high".
        - missed_low   : Precio donde se detectó un "missed pivot low".
        - ghost_level  : Precio del nivel fantasma horizontal actual.

        ChartPrime (S/R con volumen)
        ----------------------------
        - sr_res        : Precio del nivel de resistencia.
        - sr_sup        : Precio del nivel de soporte.
        - breakout_res  : Ruptura alcista de resistencia.
        - breakout_sup  : Ruptura bajista de soporte.
        - res_holds     : Resistencia rechaza el precio.
        - sup_holds     : Soporte sostiene el precio.
        - sup_is_res    : Soporte convertido en resistencia.
        - res_is_sup    : Resistencia convertida en soporte.

        Salidas y Protecciones
        ----------------------
        - breakout_res_active  : Ruptura alcista de resistencia activa (retest).
        - breakout_sup_active  : Ruptura bajista de soporte activo (retest).
        - retest_failed_res    : Estado de ruptura alcista de resistencia activa (retest).
        - retest_failed_sup    : Estado de ruptura bajista del soporte activo (retest).
    """
    df = dataframe.copy()
    small_length = pivot_length // 2

    # ---------------------------------------------------------------------------
    # 1. Volumen delta
    # ---------------------------------------------------------------------------
    # Volumen en velas alcistas menos volumen en velas bajistas
    df['delta_vol'] = np.where(
        df['close'] > df['open'], df['volume'],
        np.where(df['close'] < df['open'], -df['volume'], 0)
    )

    # Umbrales móviles
    df['vol_hi'] = (df['delta_vol'] / 2.5).rolling(vol_len).max()
    df['vol_lo'] = (df['delta_vol'] / 2.5).rolling(vol_len).min()

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
    df['pivot_os'] = np.nan
    df.loc[df['ph_big'], 'pivot_os'] = 1
    df.loc[df['pl_big'], 'pivot_os'] = 0
    df['pivot_os'] = df['pivot_os'].ffill()

    df['pivot_high'] = np.where(df['ph_big'], df['high'], np.nan)
    df['pivot_low']  = np.where(df['pl_big'], df['low'], np.nan)

    # Pivotes secundarios (para missed points)
    ph_small, pl_small = unique_pivots(df['high'], small_length)
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
    cond_sup = df['pl_big'] & (df['delta_vol'] > df['vol_hi']) # Support lvl with Positive Volume
    cond_res = df['ph_big'] & (df['delta_vol'] < df['vol_lo']) # Resistance lvl with Negative Volume

    df['sr_sup'] = np.nan
    df['sr_res'] = np.nan
    df.loc[cond_sup, 'sr_sup'] = df['low']
    df.loc[cond_res, 'sr_res'] = df['high']

    df['sr_sup'] = df['sr_sup'].ffill()
    df['sr_res'] = df['sr_res'].ffill()

    # Márgenes exteriores de la caja
    df['res_upper'] = df['sr_res'] + df['width']  # parte alta de la resistencia
    df['sup_lower'] = df['sr_sup'] - df['width']  # parte baja del soporte

    # Crear identificadores únicos para cada nivel de resistencia y soporte
    df['res_id'] = np.nan
    df.loc[cond_res, 'res_id'] = 'RES_' + df.loc[cond_res].index.astype(str)
    df['res_id'] = df['res_id'].ffill()

    df['sup_id'] = np.nan
    df.loc[cond_sup, 'sup_id'] = 'SUP_' + df.loc[cond_sup].index.astype(str)
    df['sup_id'] = df['sup_id'].ffill()

    # ---------------------------------------------------------------------------
    # 6. Eventos de ruptura
    # ---------------------------------------------------------------------------
    prev_high = df['high'].shift(1)
    prev_low  = df['low'].shift(1)
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)

    # Velas verdes y rojas
    bullish_candle = (df['close'] > df['open'])
    bearish_candle = (df['close'] < df['open'])

    def breakout_res(level: str = "res_upper") -> pd.Series:
        condition = (
            (df['low'] > df[level]) &
            (prev_low  <= df[level].shift(1))
        )
        return condition

    def breakout_sup(level: str = "sup_lower") -> pd.Series:
        condition = (
            (df['high'] < df[level]) &
            (prev_high  >= df[level].shift(1))
        )
        return condition

    # Solo primer breakouts por nivel, se desea restablecer cuando nueva aparición de S/R
    def first_occurrence_only(breakout_col, level_id_col):
        cum_breaks = df.groupby(level_id_col, dropna=True)[breakout_col].cumsum()
        return df[breakout_col] & (cum_breaks == 1)

    df['breakout_res'] = breakout_res()
    df['breakout_sup'] = breakout_sup()

    df['breakout_res'] = first_occurrence_only("breakout_res", "res_id")
    df['breakout_sup'] = first_occurrence_only("breakout_sup", "sup_id")

    # ---------------------------------------------------------------------------
    # 7. Cambios de rol
    # ---------------------------------------------------------------------------
    df['res_is_sup'] = pd.Series(dtype=bool)
    df['sup_is_res'] = pd.Series(dtype=bool)

    df.loc[df['breakout_res'], 'res_is_sup'] = True   # Resistencia -> Soporte
    df.loc[df['breakout_sup'], 'sup_is_res'] = True   # Soporte -> Resistencia

    # Invalidar cuando nuevo soporte creado
    new_sup = (df['sr_sup'] != df['sr_sup'].shift(1))
    df.loc[new_sup, 'res_is_sup'] = False
    # Invalidar cuando nueva resistencia creada
    new_res = (df['sr_res'] != df['sr_res'].shift(1))
    df.loc[new_res, 'sup_is_res'] = False

    df['res_is_sup'] = df['res_is_sup'].ffill()
    df['sup_is_res'] = df['sup_is_res'].ffill()

    df['res_active'] = np.where(df['sup_is_res'], df['sr_sup'].where(df['breakout_sup']).ffill(), df['sr_res'])
    df['sup_active'] = np.where(df['res_is_sup'], df['sr_res'].where(df['breakout_res']).ffill(), df['sr_sup'])

    df['res_active_upper'] = df['res_active'] + df['width']  # parte alta de la resistencia activa
    df['sup_active_lower'] = df['sup_active'] - df['width']  # parte baja del soporte activo

    # ---------------------------------------------------------------------------
    # 8. Eventos de rechazo
    # ---------------------------------------------------------------------------
    prev_res_active = df['res_active'].shift(1)
    prev_sup_active = df['sup_active'].shift(1)

    # Rechazo en resistencia
    valid_res = (df['sr_res'] != df['sup_active'])
    df.loc[valid_res, 'res_holds'] = (
        bearish_candle[valid_res] &
        (prev_high[valid_res] >= prev_res_active[valid_res]) &
        (prev_close[valid_res] < prev_res_active[valid_res])
    )

    # Rechazo en soporte
    valid_sup = (df['sr_sup'] != df['res_active'])
    df.loc[valid_sup, 'sup_holds'] = (
        bullish_candle[valid_sup] &
        (prev_low[valid_sup] <= prev_sup_active[valid_sup]) &
        (prev_close[valid_sup] > prev_sup_active[valid_sup])
    )

    # ---------------------------------------------------------------------------
    # 9. Protecciones
    # ---------------------------------------------------------------------------
    df['breakout_res_active'] = breakout_res("res_active_upper")
    df['breakout_sup_active'] = breakout_sup("sup_active_lower")
    df['breakout_res_active'] = first_occurrence_only("breakout_res_active", "res_id")
    df['breakout_sup_active'] = first_occurrence_only("breakout_sup_active", "sup_id")

    df['retest_failed_res'] = pd.Series(dtype=bool)
    df['retest_failed_sup'] = pd.Series(dtype=bool)

    # Fallo de rebote en retesteo, esperar...
    df.loc[(df['breakout_res_active'] & df['sup_is_res']), 'retest_failed_res'] = True
    df.loc[(df['breakout_sup_active'] & df['res_is_sup']), 'retest_failed_sup'] = True

    # Invalidar cuando nuevo soporte o resistencia creado
    df.loc[new_sup | new_res, 'retest_failed_sup'] = False
    df.loc[new_res | new_sup, 'retest_failed_res'] = False

    df['retest_failed_sup'] = df['retest_failed_sup'].ffill()
    df['retest_failed_res'] = df['retest_failed_res'].ffill()

    # ---------------------------------------------------------------------------
    # 10. Limpieza
    # ---------------------------------------------------------------------------
    cols_to_drop = [
        'delta_vol', 'vol_hi', 'vol_lo', 'atr', 'width',
        'ph_big', 'pl_big', 'ph_small', 'pl_small', 'pivot_big_event', 'segment_id',
        'prev_seg_max', 'prev_seg_min', 'prev_seg_max_idx', 'prev_seg_min_idx',
        # 'missed_high', 'missed_low', 'res_active', 'sup_active',
        'sup_lower', 'res_upper', 'sup_active_lower', 'res_active_upper',
        'res_id', 'sup_id'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df

def fibonacci(last_swing_high_val, last_swing_low_val,
              lvl = 0.618, high = False, low = False):
    """Cálculo de niveles Fibo en el pullback"""
    diff = last_swing_high_val - last_swing_low_val
    price = last_swing_high_val - (diff * lvl)
    return high > price if high else low > price
