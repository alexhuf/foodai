import pandas as pd
import requests

print("📡 Connecting to Open-Meteo Historical Archive...")

# 1. Load the dataset
try:
    df = pd.read_csv('Ultimate_Super_Dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("ERROR: Ultimate_Super_Dataset.csv not found!")
    exit()

# 2. Get the date range for the API
min_date = df['date'].min().strftime('%Y-%m-%d')
max_date = df['date'].max().strftime('%Y-%m-%d')
print(f"Requesting Okemos, MI weather data from {min_date} to {max_date}...")

# 3. Call the Free API (Okemos Lat: 42.72, Lon: -84.40)
url = f"https://archive-api.open-meteo.com/v1/archive?latitude=42.7244&longitude=-84.4028&start_date={min_date}&end_date={max_date}&daily=temperature_2m_max,daylight_duration&timezone=America%2FNew_York"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    daily = data['daily']
    
    # Create a temporary dataframe with the exact real data
    weather_df = pd.DataFrame({
        'date': pd.to_datetime(daily['time']),
        'real_max_temp_c': daily['temperature_2m_max'],
        'real_daylight_seconds': daily['daylight_duration']
    })
    
    # Convert formats
    weather_df['Real_High_Temp'] = (weather_df['real_max_temp_c'] * 9/5) + 32
    weather_df['Real_Daylight'] = weather_df['real_daylight_seconds'] / 3600
    
    # Re-calculate our ML boolean flags based on the REAL data
    weather_df['Is_Freezing'] = (weather_df['Real_High_Temp'] <= 32).astype(str)
    weather_df['Is_Hot'] = (weather_df['Real_High_Temp'] >= 75).astype(str)
    
    # 4. Overwrite the old approximated columns
    cols_to_replace = ['Approx_High_Temp', 'Daylight_Hours', 'Is_Freezing', 'Is_Hot']
    df = df.drop(columns=[c for c in cols_to_replace if c in df.columns])
    
    df = pd.merge(df, weather_df[['date', 'Real_High_Temp', 'Real_Daylight', 'Is_Freezing', 'Is_Hot']], on='date', how='left')
    
    # Rename them back so the build_brain.py script recognizes them
    df.rename(columns={'Real_High_Temp': 'Approx_High_Temp', 'Real_Daylight': 'Daylight_Hours'}, inplace=True)
    
    # Fill any missing days with the previous day's weather
    df.ffill(inplace=True)
    
    # Save the upgraded dataset
    df.to_csv('Ultimate_Super_Dataset.csv', index=False)
    print("✅ SUCCESS! Your dataset has been upgraded with real atmospheric data.")
    
else:
    print(f"❌ API Failed. Status code: {response.status_code}")