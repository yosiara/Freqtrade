# custom_indicators.py
"""
Indicadores técnicos personalizados para Freqtrade.
Incluye detección de pivotes con niveles fantasma (LuxAlgo) y soportes/resistencias
basados en volumen delta (ChartPrime), todo en una sola pasada eficiente.
"""

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

        LuxAlgo
        -------
        - pivot_high   : Precio del pivote alto confirmado.
        - pivot_low    : Precio del pivote bajo confirmado.
        - pivot_os     : Estado del último pivote (1=alto, 0=bajo, -1=inicial).
        - ghost_level  : Nivel fantasma horizontal actual (precio).
        - missed_high  : Precio donde se detectó un "missed pivot high".
        - missed_low   : Precio donde se detectó un "missed pivot low".

        ChartPrime (S/R con volumen)
        ----------------------------
        - sr_support    : Nivel de soporte activo (basado en pivote bajo + volumen).
        - sr_resistance : Nivel de resistencia activo (basado en pivote alto + volumen).
        - breakout_res  : Ruptura alcista de resistencia.
        - res_holds     : Resistencia rechaza el precio.
        - sup_holds     : Soporte sostiene el precio.
        - breakout_sup  : Ruptura bajista de soporte.
        - res_is_sup    : Resistencia previa actúa como soporte (cambio de rol).
        - sup_is_res    : Soporte previo actúa como resistencia (cambio de rol).
    """
    df = dataframe.copy()
    n = len(df)

    # -------------------------------------------------------------------------
    # Extraer arrays para rendimiento
    # -------------------------------------------------------------------------
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_ = df['open'].values
    volume = df['volume'].values

    # -------------------------------------------------------------------------
    # 1. Cálculos vectorizados previos al bucle
    # -------------------------------------------------------------------------
    # Volumen delta: volumen en velas alcistas menos volumen en velas bajistas
    delta_vol = np.where(close > open_, volume,
                         np.where(close < open_, -volume, 0))

    # Umbrales móviles del volumen delta
    vol_hi = pd.Series(delta_vol).rolling(vol_len).max().values
    vol_lo = pd.Series(delta_vol).rolling(vol_len).min().values

    # ATR para el ancho de las zonas S/R (usando talib para precisión)
    atr = ta.ATR(df, timeperiod=200).values
    width = atr * box_width

    # -------------------------------------------------------------------------
    # 2. Detección vectorizada de pivotes (ventana centrada)
    # -------------------------------------------------------------------------
    half = pivot_length
    rolling_high = pd.Series(high).rolling(window=2 * half + 1, center=True, min_periods=1)
    rolling_low = pd.Series(low).rolling(window=2 * half + 1, center=True, min_periods=1)

    # Un pivote alto es el máximo de su ventana y es único
    is_ph = (high == rolling_high.max().values) & (
        rolling_high.apply(lambda x: np.sum(x == np.max(x)) == 1, raw=True).values.astype(bool)
    )
    # Un pivote bajo es el mínimo de su ventana y es único
    is_pl = (low == rolling_low.min().values) & (
        rolling_low.apply(lambda x: np.sum(x == np.min(x)) == 1, raw=True).values.astype(bool)
    )

    # -------------------------------------------------------------------------
    # 3. Bucle principal para estado LuxAlgo (máximos/mínimos rodantes y missed levels)
    # -------------------------------------------------------------------------
    # Arrays de salida
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)
    os_arr = np.full(n, -1, dtype=int)
    ghost_level = np.full(n, np.nan)
    missed_high = np.full(n, np.nan)
    missed_low = np.full(n, np.nan)

    # Variables de estado persistentes (estilo Pine Script)
    max_val = 0.0
    min_val = 0.0
    max_x1 = 0
    min_x1 = 0
    follow_max = 0.0
    follow_max_x1 = 0
    follow_min = 0.0
    follow_min_x1 = 0
    os = 0
    px1 = 0
    py1 = 0.0
    ghost = np.nan

    # Solo evaluamos dentro del rango donde la ventana centrada tiene datos completos
    for i in range(half, n - half):
        lookback = i - half
        high_lb = high[lookback]
        low_lb = low[lookback]

        # Actualización de máximos/mínimos rodantes (se alimentan con el precio de hace 'half' barras)
        if high_lb > max_val:
            max_val = high_lb
            max_x1 = lookback
            follow_min = low_lb
        if low_lb < min_val:
            min_val = low_lb
            min_x1 = lookback
            follow_max = high_lb

        if low_lb < follow_min:
            follow_min = low_lb
            follow_min_x1 = lookback
        if high_lb > follow_max:
            follow_max = high_lb
            follow_max_x1 = lookback

        # --- Procesar pivote alto ---
        if is_ph[i]:
            pivot_high[i] = high[i]
            if os == 1:
                missed_low[min_x1] = min_val
                px1, py1 = min_x1, min_val
                ghost = min_val
            elif high[i] < max_val:
                missed_high[max_x1] = max_val
                missed_low[follow_min_x1] = follow_min
                px1, py1 = max_x1, max_val
                ghost = max_val
                px1, py1 = follow_min_x1, follow_min
                ghost = follow_min

            os = 1
            max_val = high[i]
            min_val = high[i]
            px1, py1 = i, high[i]

        # --- Procesar pivote bajo ---
        if is_pl[i]:
            pivot_low[i] = low[i]
            if os == 0:
                missed_high[max_x1] = max_val
                px1, py1 = max_x1, max_val
                ghost = max_val
            elif low[i] > min_val:
                missed_high[follow_max_x1] = follow_max
                missed_low[min_x1] = min_val
                px1, py1 = min_x1, min_val
                ghost = min_val
                px1, py1 = follow_max_x1, follow_max
                ghost = follow_max

            os = 0
            max_val = low[i]
            min_val = low[i]
            px1, py1 = i, low[i]

        os_arr[i] = os
        ghost_level[i] = ghost

    # Proyección final: nivel fantasma no confirmado hasta el último precio
    if px1 < n - 1:
        if os == 1:
            segment = low[px1 + 1:]
            if len(segment) > 0:
                min_idx = np.argmin(segment) + px1 + 1
                missed_low[min_idx] = segment.min()
        else:
            segment = high[px1 + 1:]
            if len(segment) > 0:
                max_idx = np.argmax(segment) + px1 + 1
                missed_high[max_idx] = segment.max()

    # -------------------------------------------------------------------------
    # 4. ChartPrime: Asignación de S/R basada en pivotes + filtro de volumen
    # -------------------------------------------------------------------------
    support_series = pd.Series(np.nan, index=df.index)
    resistance_series = pd.Series(np.nan, index=df.index)

    # Condiciones: pivote bajo con delta_vol > umbral superior => soporte
    #              pivote alto con delta_vol < umbral inferior => resistencia
    cond_sup = is_pl & (delta_vol > vol_hi)
    cond_res = is_ph & (delta_vol < vol_lo)

    support_series.loc[cond_sup] = low[cond_sup]
    resistance_series.loc[cond_res] = high[cond_res]

    # Rellenar hacia adelante para tener el nivel activo en cada vela
    support_series = support_series.ffill()
    resistance_series = resistance_series.ffill()

    # Niveles ajustados por el ancho de la caja (para detectar rupturas)
    sup_level_1 = support_series - width
    res_level_1 = resistance_series + width

    # -------------------------------------------------------------------------
    # 5. Eventos de ruptura/rechazo (ChartPrime) - Vectorizado
    # -------------------------------------------------------------------------
    # Convertir arrays a Series con el índice del DataFrame para usar shift()
    low_s = pd.Series(low, index=df.index)
    high_s = pd.Series(high, index=df.index)
    close_s = pd.Series(close, index=df.index)

    # Ruptura alcista de resistencia
    breakout_res = (low_s > res_level_1) & (low_s.shift(1) <= res_level_1.shift(1))
    # Rechazo en resistencia
    res_holds = (high_s >= resistance_series) & (close_s < resistance_series)
    # Sostenimiento en soporte
    sup_holds = (low_s <= support_series) & (close_s > support_series)
    # Ruptura bajista de soporte
    breakout_sup = (high_s < sup_level_1) & (high_s.shift(1) >= sup_level_1.shift(1))

    # -------------------------------------------------------------------------
    # 6. Cambio de rol (Resistencia -> Soporte y viceversa)
    # -------------------------------------------------------------------------
    res_is_sup_arr = np.full(n, False)
    sup_is_res_arr = np.full(n, False)

    res_is_sup = False
    sup_is_res = False
    for i in range(n):
        if breakout_res.iat[i]:
            res_is_sup = True
        elif res_holds.iat[i]:
            res_is_sup = False

        if breakout_sup.iat[i]:
            sup_is_res = True
        elif sup_holds.iat[i]:
            sup_is_res = False

        res_is_sup_arr[i] = res_is_sup
        sup_is_res_arr[i] = sup_is_res

    # -------------------------------------------------------------------------
    # 7. Asignación final al DataFrame
    # -------------------------------------------------------------------------
    df['pivot_high'] = pivot_high
    df['pivot_low'] = pivot_low
    df['pivot_os'] = os_arr
    df['ghost_level'] = ghost_level
    df['missed_high'] = missed_high
    df['missed_low'] = missed_low

    df['sr_support'] = support_series.values
    df['sr_resistance'] = resistance_series.values
    df['breakout_res'] = breakout_res.values
    df['res_holds'] = res_holds.values
    df['sup_holds'] = sup_holds.values
    df['breakout_sup'] = breakout_sup.values
    df['res_is_sup'] = res_is_sup_arr
    df['sup_is_res'] = sup_is_res_arr

    return df
