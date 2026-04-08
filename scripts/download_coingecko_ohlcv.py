import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os

def download_coingecko_ohlcv(coin_id, vs_currency, days, timeframe_minutes):
    # Calcula la fecha de inicio
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': vs_currency,
        'from': start_timestamp,
        'to': end_timestamp
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'prices' not in data:
        print("Error: No se pudieron descargar los datos.")
        return None
    
    # Crea el DataFrame de OHLCV
    ohlcv_data = []
    prices = data['prices']
    volumes = data['total_volumes']
    
    # CoinGecko da datos en intervalos variables, se debe remuestrear al timeframe deseado
    # Se crea un DataFrame con los precios y volúmenes
    df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
    
    df = pd.merge(df_prices, df_volumes, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Se remuestrea al timeframe deseado (en minutos)
    ohlc = df['price'].resample(f'{timeframe_minutes}T').ohlc()
    ohlc['volume'] = df['volume'].resample(f'{timeframe_minutes}T').sum()
    
    ohlc = ohlc.dropna()
    ohlc.reset_index(inplace=True)
    ohlc.rename(columns={
        'timestamp': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close'
    }, inplace=True)
    
    # Formatea la fecha como string
    ohlc['date'] = ohlc['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return ohlc

# --- Configuración ---
coin_id = 'bitcoin'  # Cambia según la moneda
vs_currency = 'usd'
days_to_download = 95
timeframe = 5  # en minutos, e.g., 60 para 1h

# --- Ejecución ---
print(f"Descargando datos de {coin_id} para los últimos {days_to_download} días...")
data = download_coingecko_ohlcv(coin_id, vs_currency, days_to_download, timeframe)

if data is not None:
    # Guarda en un archivo CSV
    csv_filename = f"{coin_id}_ohlcv_{timeframe}m.csv"
    data.to_csv(csv_filename, index=False)
    print(f"Datos guardados en {csv_filename}")
else:
    print("No se pudo descargar la data.")