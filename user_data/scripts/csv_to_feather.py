import os

import pandas as pd


# ===================== Configuración =========================
coin_id = 'bitcoin'
pair = 'BTC_USDT'
timeframe = '5m'
from_path = f'data/coingecko/{coin_id}_ohlcv_5m.csv'
to_path = f'data/coinex/{pair}-{timeframe}.feather'
# =============================================================

# Lee el CSV que generaste
df = pd.read_csv(from_path)
df['date'] = pd.to_datetime(df['date'])

# Asegura que tenga el orden de columnas correcto: date, open, high, low, close, volume
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

# Guarda en formato feather
df.reset_index(drop=True).to_feather(to_path)
print(f"Archivo convertido y guardado en: {to_path}")
