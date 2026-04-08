import pandas as pd
import os

# Lee el CSV que generaste
df = pd.read_csv('bitcoin_ohlcv_5m.csv')
df['date'] = pd.to_datetime(df['date'])

# Asegura que tenga el orden de columnas correcto: date, open, high, low, close, volume
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

# Define la ruta donde Freqtrade espera los datos
# Ejemplo: data/coinex/BTC_USDT-1h.feather
pair = 'BTC_USDT'  # Nombre del par
timeframe = '5m'
data_path = f'data/coinex/{pair}-{timeframe}.feather'

# Guarda en formato feather
df.reset_index(drop=True).to_feather(data_path)
print(f"Archivo convertido y guardado en: {data_path}")