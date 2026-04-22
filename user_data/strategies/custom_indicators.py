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
        - ghost_level  : Precio del nivel fantasma horizontal actual.

        ChartPrime (S/R con volumen)
        ----------------------------
        - sr_res        : Precio del nivel de resistencia.
        - sr_sup        : Precio del nivel de soporte.
        - breakout_res  : Ruptura alcista de resistencia.
        - breakout_sup  : Ruptura bajista de soporte.
        - res_holds     : Resistencia rechaza el precio.
        - sup_holds     : Soporte sostiene el precio.
        - res_active    : Soporte puede convertirse en resistencia potencial.
        - sup_active    : Resistencia puede convertirse en soporte potencial.

        Salidas y Protecciones
        ----------------------
        - breakout_res_active  : Ruptura alcista de resistencia activa (retest).
        - breakout_sup_active  : Ruptura bajista de soporte activo (retest).
        - retest_failed_res    : Estado de ruptura alcista de resistencia activa (retest).
        - retest_failed_sup    : Estado de ruptura bajista del soporte activo (retest).
    """
    df = dataframe.copy()

    # ---------------------------------------------------------------------------
    # 1. Volumen delta
    # ---------------------------------------------------------------------------
    df['delta_vol'] = np.where(
        df['close'] > df['open'], df['volume'],
        np.where(df['close'] < df['open'], -df['volume'], 0)
    ) # Volumen en velas alcistas menos volumen en velas bajistas

    # Umbrales móviles
    df['vol_hi'] = (df['delta_vol'] / 2.5).rolling(vol_len).max()
    df['vol_lo'] = (df['delta_vol'] / 2.5).rolling(vol_len).min()

    # ATR para el ancho de las zonas S/R
    df['atr'] = ta.ATR(df, timeperiod=200)
    df['width'] = df['atr'] * box_width

    # ---------------------------------------------------------------------------
    # 2. Pivotes
    # ---------------------------------------------------------------------------
    def pivot_high(series: pd.Series, length: int) -> pd.Series:
        half = length
        roll = series.rolling(window=2 * half + 1, center=True)
        max_vals = roll.max()
        is_max = (series == max_vals)
        unique_max = roll.apply(lambda x: np.sum(x == x.max()) == 1, raw=True)
        return is_max & unique_max

    def pivot_low(series: pd.Series, length: int) -> pd.Series:
        half = length
        roll = series.rolling(window=2 * half + 1, center=True)
        min_vals = roll.min()
        is_min = (series == min_vals)
        unique_min = roll.apply(lambda x: np.sum(x == x.min()) == 1, raw=True)
        return is_min & unique_min

    df['cond_ph'] = pivot_high(df['high'], pivot_length)
    df['cond_pl'] = pivot_low(df['low'], pivot_length)
    df['pivot_high'] = np.where(df['cond_ph'], df['high'], np.nan)
    df['pivot_low']  = np.where(df['cond_pl'], df['low'], np.nan)

    # Estado
    df['pivot_os'] = np.nan
    df.loc[df['cond_ph'], 'pivot_os'] = 1
    df.loc[df['cond_pl'], 'pivot_os'] = 0
    df['pivot_os'] = df['pivot_os'].ffill()

    # ---------------------------------------------------------------------------
    # 3. Missed Levels
    # ---------------------------------------------------------------------------
    df['segment_id'] = (df['cond_ph'] | df['cond_pl']).cumsum()

    group_seg_hi = df.groupby('segment_id')['high']
    group_seg_lo = df.groupby('segment_id')['low']

    df['prev_seg_max'] = group_seg_hi.transform('max').shift(1)
    df['prev_seg_min'] = group_seg_lo.transform('min').shift(1)
    df['prev_seg_max_idx'] = group_seg_hi.transform('idxmax').shift(1)
    df['prev_seg_min_idx'] = group_seg_lo.transform('idxmin').shift(1)

    cond_missed_hi = df['cond_pl'] & (df['prev_seg_max'] > df['low'])
    cond_missed_lo = df['cond_ph'] & (df['prev_seg_min'] < df['high'])

    df['missed_high'] = np.nan
    df['missed_low']  = np.nan

    # Filtros de no coincidencia con pivotes
    if cond_missed_hi.any():
        target_idx = df.loc[cond_missed_hi, 'prev_seg_max_idx'].dropna().astype(int)
        target_vals = df.loc[cond_missed_hi, 'prev_seg_max'].values
        is_pivot = df.loc[target_idx, 'cond_ph'].values | df.loc[target_idx, 'cond_pl'].values
        valid = ~is_pivot
        df.loc[target_idx[valid], 'missed_high'] = target_vals[valid]

    if cond_missed_lo.any():
        target_idx = df.loc[cond_missed_lo, 'prev_seg_min_idx'].dropna().astype(int)
        target_vals = df.loc[cond_missed_lo, 'prev_seg_min'].values
        is_pivot = df.loc[target_idx, 'cond_ph'].values | df.loc[target_idx, 'cond_pl'].values
        valid = ~is_pivot
        df.loc[target_idx[valid], 'missed_low'] = target_vals[valid]

    # Ghost level
    last_missed = df['missed_high'].combine_first(df['missed_low'])
    df['ghost_level'] = last_missed.ffill()

    # ---------------------------------------------------------------------------
    # 4. ChartPrime S/R
    # ---------------------------------------------------------------------------
    cond_res = df['cond_ph'] & (df['delta_vol'] < df['vol_lo']) # Resistance lvl with Negative Volume
    cond_sup = df['cond_pl'] & (df['delta_vol'] > df['vol_hi']) # Support lvl with Positive Volume

    df['sr_res'] = np.nan
    df['sr_sup'] = np.nan
    df.loc[cond_res, 'sr_res'] = df['high']
    df.loc[cond_sup, 'sr_sup'] = df['low']

    df['sr_res'] = df['sr_res'].ffill()
    df['sr_sup'] = df['sr_sup'].ffill()

    # Márgenes exteriores de la caja
    df['res_upper'] = df['sr_res'] + df['width']  # parte alta de la resistencia
    df['sup_lower'] = df['sr_sup'] - df['width']  # parte baja del soporte

    # IDs únicos para cada nivel
    df['res_id'] = np.where(cond_res, ('RES_' + df.index.astype(str)), np.nan)
    df['res_id'] = df['res_id'].ffill()
    df['sup_id'] = np.where(cond_sup, ('SUP_' + df.index.astype(str)), np.nan)
    df['sup_id'] = df['sup_id'].ffill()

    # -------------------------------------------------------------------------
    # 5. Variables y funciones auxiliares
    # -------------------------------------------------------------------------
    prev_high = df['high'].shift(1)
    prev_low  = df['low'].shift(1)
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)

    # Velas verdes y rojas
    bullish_candle = (df['close'] > df['open'])
    bearish_candle = (df['close'] < df['open'])

    # Nuevos niveles
    new_res = (df['res_id'] != df['res_id'].shift(1))
    new_sup = (df['sup_id'] != df['sup_id'].shift(1))

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

    def first_breakout(breakout_col: str, level_id_col: str) -> pd.Series:
        cum_breaks = df.groupby(level_id_col, dropna=True)[breakout_col].cumsum()
        return df[breakout_col] & (cum_breaks == 1)

    # ---------------------------------------------------------------------------
    # 6. Eventos de ruptura
    # ---------------------------------------------------------------------------
    df['cycle_res'] = new_res.cumsum()
    df['cycle_sup'] = new_sup.cumsum()
    df['group_res'] = df['res_id'] + '_' + df['cycle_sup'].astype(str)
    df['group_sup'] = df['sup_id'] + '_' + df['cycle_res'].astype(str)

    df['breakout_res'] = breakout_res()
    df['breakout_sup'] = breakout_sup()

    df['breakout_res'] = first_breakout("breakout_res", "group_res")
    df['breakout_sup'] = first_breakout("breakout_sup", "group_sup")

    # ---------------------------------------------------------------------------
    # 7. Cambios de rol
    # ---------------------------------------------------------------------------
    df['res_active'] = np.nan
    df['sup_active'] = np.nan

    # 1. Inicialización con niveles originales
    df.loc[new_res, 'res_active'] = df['sr_res']
    df.loc[new_sup, 'sup_active'] = df['sr_sup']

    # 2. Ruptura de resistencia -> se convierte en soporte
    df.loc[df['breakout_res'], 'sup_active'] = df['sr_res']
    # Al convertirse en soporte, la resistencia activa debe volver atrás
    df.loc[df['breakout_res'], 'res_active'] = df['sr_res']

    # 3. Ruptura de soporte -> se convierte en resistencia
    df.loc[df['breakout_sup'], 'res_active'] = df['sr_sup']
    # Al convertirse en resistencia, el soporte activo debe volver atrás
    df.loc[df['breakout_sup'], 'sup_active'] = df['sr_sup']

    # 4. Propagación hacia adelante
    df['res_active'] = df['res_active'].ffill()
    df['sup_active'] = df['sup_active'].ffill()

    # 5. Márgenes de los niveles activos
    df['res_active_upper'] = df['res_active'] + df['width']
    df['sup_active_lower'] = df['sup_active'] - df['width']

    # ---------------------------------------------------------------------------
    # 8. Eventos de rechazo
    # ---------------------------------------------------------------------------
    prev_res_active = df['res_active'].shift(1)
    prev_sup_active = df['sup_active'].shift(1)

    # Rechazo en resistencia
    valid_res = (df['sr_res'] != df['sup_active'])  # Not flipped support
    df.loc[valid_res, 'res_holds'] = (
        bearish_candle[valid_res] &
        (prev_high[valid_res] >= prev_res_active[valid_res]) &
        (prev_close[valid_res] < prev_res_active[valid_res])
    )

    # Rechazo en soporte
    valid_sup = (df['sr_sup'] != df['res_active'])  # Not flipped resistence
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
    df['breakout_res_active'] = first_breakout("breakout_res_active", "group_res")
    df['breakout_sup_active'] = first_breakout("breakout_sup_active", "group_sup")

    df['retest_failed_res'] = pd.Series(dtype='boolean')
    df['retest_failed_sup'] = pd.Series(dtype='boolean')

    # Fallo de rebote en retesteo
    df.loc[(df['breakout_res_active'] & ~valid_sup), 'retest_failed_res'] = True
    df.loc[(df['breakout_sup_active'] & ~valid_res), 'retest_failed_sup'] = True

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
        'cond_ph', 'cond_pl', 'segment_id',
        'prev_seg_max', 'prev_seg_min', 'prev_seg_max_idx', 'prev_seg_min_idx',
        'missed_high', 'missed_low',
        'sup_lower', 'res_upper', 'sup_active_lower', 'res_active_upper',
        'cycle_res', 'cycle_sup', 'group_res', 'group_sup'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df

def fibonacci(last_swing_high_val, last_swing_low_val,
              lvl = 0.618, high = False, low = False):
    """Cálculo de niveles Fibo en el pullback"""
    diff = last_swing_high_val - last_swing_low_val
    price = last_swing_high_val - (diff * lvl)
    return high > price if high else low > price
