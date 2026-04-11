import pandas as pd
import numpy as np
import requests
import json
import warnings
warnings.filterwarnings('ignore')

print("🧬 INITIALIZING V15.1 MASTER TELEMETRY LEDGER (HYPER-GRANULAR FUSION)...")

# ==============================================================================
# --- CONFIGURATION: THE 7 CORE FILES ---
# ==============================================================================
UBER_EATS_FILE      = "v5_omni_matrix.csv"
SAMSUNG_WEIGHT_FILE = "com.samsung.health.weight.20260411021045.csv"
NOOM_ACTIONS_FILE   = "actions.csv" 
HR_FILE             = "com.samsung.shealth.tracker.heart_rate.20260411021045.csv"
STRESS_FILE         = "com.samsung.shealth.stress.20260411021045.csv"
STEPS_FILE          = "com.samsung.shealth.tracker.pedometer_step_count.20260411021045.csv"
CALS_FILE           = "com.samsung.shealth.calories_burned.details.20260411021045.csv"
# ==============================================================================

def load_and_resample(filepath, time_col, data_cols, freq='15min', agg_dict=None):
    try:
        # BUG FIX: index_col=False prevents Samsung's leading commas from shifting the headers
        df = pd.read_csv(filepath, skiprows=1, index_col=False)
        if time_col not in df.columns:
            possible_times = ['start_time', 'create_time', 'update_time', 'com.samsung.health.step_count.start_time', 'com.samsung.health.heart_rate.start_time']
            time_col = next((pt for pt in possible_times if pt in df.columns and df[pt].notna().any()), None)
            if not time_col: return pd.DataFrame()

        df = df.dropna(subset=[time_col])
        clean_time = df[time_col].astype(str).str.replace(r'([-+]\d{2}:?\d{2}|Z)$', '', regex=True)
        df['datetime'] = pd.to_datetime(clean_time, errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
        
        if agg_dict is None: agg_dict = {col: 'mean' for col in data_cols}
        return df[data_cols].resample(freq).agg(agg_dict)
    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")
        return pd.DataFrame()

# --- 1. SENSOR FUSION: DUAL-WEIGHT ENGINE (SAMSUNG + NOOM) ---
print("Fusing Samsung & Noom Weight Logs...")
try:
    df_sam = pd.read_csv(SAMSUNG_WEIGHT_FILE, skiprows=1, index_col=False)
    clean_time_sam = df_sam['create_time'].astype(str).str.replace(r'([-+]\d{2}:?\d{2}|Z)$', '', regex=True)
    df_sam['date'] = pd.to_datetime(clean_time_sam, errors='coerce').dt.floor('D')
    df_sam_wt = df_sam.groupby('date').agg({'weight': 'mean'}).rename(columns={'weight': 'sam_weight'})
except Exception as e:
    print(f"⚠️ Could not load Samsung weight. Error: {e}")
    df_sam_wt = pd.DataFrame(columns=['sam_weight'])

try:
    df_actions = pd.read_csv(NOOM_ACTIONS_FILE)
    df_weigh_in = df_actions[df_actions['actionType'] == 'WEIGH_IN'].copy()
    
    def extract_weight(json_str):
        try:
            return json.loads(json_str).get('weightInKg', np.nan) * 2.20462
        except:
            return np.nan
            
    df_weigh_in['noom_weight'] = df_weigh_in['jsonString'].apply(extract_weight)
    df_weigh_in['date'] = pd.to_datetime(df_weigh_in['date'], errors='coerce').dt.floor('D')
    df_noom_wt = df_weigh_in.groupby('date').agg({'noom_weight': 'mean'})
except Exception as e:
    print(f"⚠️ Could not load Noom actions data. Error: {e}")
    df_noom_wt = pd.DataFrame(columns=['noom_weight'])
    df_noom_wt.index.name = 'date'

df_weight = df_sam_wt.join(df_noom_wt, how='outer')
if not df_weight.empty:
    df_weight['true_weight'] = df_weight[['sam_weight', 'noom_weight']].mean(axis=1)
else:
    df_weight['true_weight'] = np.nan

weight_cols_to_merge = []
if not df_weight.empty and df_weight['true_weight'].notna().any():
    full_days = pd.date_range(start=df_weight.index.min(), end=df_weight.index.max(), freq='D')
    df_weight = df_weight.reindex(full_days).ffill() 
    
    windows = [3, 5, 7, 14, 30]
    weight_cols_to_merge.append('true_weight')
    
    for w in windows:
        ema_col = f'weight_ema_{w}d'
        vel_col = f'weight_velocity_{w}d'
        df_weight[ema_col] = df_weight['true_weight'].ewm(span=w, adjust=False).mean()
        df_weight[vel_col] = df_weight[ema_col].diff()
        weight_cols_to_merge.extend([ema_col, vel_col])

# --- 2. HIGH-FREQUENCY TELEMETRY (15-Min) ---
print("Processing Raw Telemetry Tensors...")
df_hr = load_and_resample(HR_FILE, "com.samsung.health.heart_rate.start_time", 
                          ['com.samsung.health.heart_rate.heart_rate'], agg_dict={'com.samsung.health.heart_rate.heart_rate':'mean'})
if not df_hr.empty: df_hr.columns = ['hr_mean']

df_stress = load_and_resample(STRESS_FILE, "create_time", ['score'])
if not df_stress.empty: df_stress.columns = ['stress_score']

df_steps = load_and_resample(STEPS_FILE, "com.samsung.health.step_count.start_time", 
                             ['com.samsung.health.step_count.calorie', 'com.samsung.health.step_count.distance'],
                             agg_dict={'com.samsung.health.step_count.calorie': 'sum', 'com.samsung.health.step_count.distance': 'sum'})
if not df_steps.empty: df_steps.columns = ['step_calories', 'step_distance']

# --- 3. METABOLIC BASELINES ---
try:
    df_cals = pd.read_csv(CALS_FILE, skiprows=1, index_col=False)
    clean_time_cals = df_cals['com.samsung.shealth.calories_burned.create_time'].astype(str).str.replace(r'([-+]\d{2}:?\d{2}|Z)$', '', regex=True)
    df_cals['date'] = pd.to_datetime(clean_time_cals, errors='coerce').dt.floor('D')
    df_cals = df_cals.groupby('date').agg({'com.samsung.shealth.calories_burned.rest_calorie': 'max'})
    df_cals.columns = ['daily_bmr']
except Exception as e:
    print(f"⚠️ Could not load Metabolic Baselines. Error: {e}")
    df_cals = pd.DataFrame(columns=['daily_bmr'])

# --- 4. BUILD THE MASTER 15-MINUTE TIMELINE ---
print("Constructing 15-Minute Epoch Grid...")
valid_mins = [df.index.min() for df in [df_hr, df_stress, df_steps] if not df.empty and pd.notnull(df.index.min())]
valid_mins.append(pd.to_datetime('2024-01-01')) 

valid_maxs = [df.index.max() for df in [df_hr, df_stress, df_steps] if not df.empty and pd.notnull(df.index.max())]
valid_maxs.append(pd.to_datetime('today'))      

min_time = min(valid_mins)
max_time = max(valid_maxs)

master_index = pd.date_range(start=min_time, end=max_time, freq='15min', name='datetime')
master_df = pd.DataFrame(index=master_index)

if not df_hr.empty: master_df = master_df.join(df_hr)
else: master_df['hr_mean'] = np.nan

if not df_stress.empty: master_df = master_df.join(df_stress)
else: master_df['stress_score'] = np.nan

if not df_steps.empty: master_df = master_df.join(df_steps)
else:
    master_df['step_calories'] = 0.0
    master_df['step_distance'] = 0.0

master_df['date_floor'] = master_df.index.floor('D')

if not df_weight.empty and weight_cols_to_merge:
    master_df = master_df.merge(df_weight[weight_cols_to_merge], left_on='date_floor', right_index=True, how='left')
else:
    for col in ['true_weight', 'weight_ema_3d', 'weight_velocity_3d', 'weight_ema_5d', 'weight_velocity_5d', 'weight_ema_7d', 'weight_velocity_7d', 'weight_ema_14d', 'weight_velocity_14d', 'weight_ema_30d', 'weight_velocity_30d']:
        master_df[col] = np.nan
        
if not df_cals.empty:
    master_df = master_df.merge(df_cals, left_on='date_floor', right_index=True, how='left')
else:
    master_df['daily_bmr'] = 1800.0 

master_df = master_df.drop(columns=['date_floor'])

# Imputations & Calculations
master_df['hr_mean'] = master_df['hr_mean'].interpolate(method='time', limit=8)
master_df['stress_score'] = master_df['stress_score'].interpolate(method='time', limit=8)
master_df[['step_calories', 'step_distance']] = master_df[['step_calories', 'step_distance']].fillna(0)

master_df['bmr_15min'] = master_df['daily_bmr'] / 96.0
master_df['total_burn_15min'] = master_df['bmr_15min'].fillna(0) + master_df['step_calories']
master_df['cumulative_daily_burn'] = master_df.groupby(master_df.index.date)['total_burn_15min'].cumsum()

# --- 5. OPEN-METEO WEATHER FETCH ---
print("Fetching Environmental Context from Open-Meteo...")
lat, lon = 42.723, -84.400 # Okemos
start_date, end_date = master_df.index.min().strftime('%Y-%m-%d'), master_df.index.max().strftime('%Y-%m-%d')
try:
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation&temperature_unit=fahrenheit"
    w_data = requests.get(weather_url).json()
    df_w = pd.DataFrame({'datetime': pd.to_datetime(w_data['hourly']['time']), 'temp_f': w_data['hourly']['temperature_2m'], 'precip': w_data['hourly']['precipitation']})
    df_w = df_w.set_index('datetime').resample('15min').ffill()
    master_df = master_df.join(df_w)
except Exception as e:
    print(f"Weather Fetch Failed: {e}")
    master_df['temp_f'], master_df['precip'] = np.nan, np.nan

# --- 6. UBER EATS MEAL FUSION (TIMEZONE FIX) ---
print("Dropping Meals into Temporal Ledger...")
try:
    df_meals = pd.read_csv(UBER_EATS_FILE)
    meal_col = 'Sequence_Target_String' if 'Sequence_Target_String' in df_meals.columns else df_meals.select_dtypes(include=['object']).columns[0]

    clean_meal_time = df_meals['Exact_Time'].astype(str).str.extract(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})')[0]
    df_meals['datetime'] = pd.to_datetime(clean_meal_time, errors='coerce')
    df_meals['datetime_15min'] = df_meals['datetime'].dt.round('15min')

    df_meals_agg = df_meals.groupby('datetime_15min').agg({
        meal_col: lambda x: ' + '.join(x.dropna().astype(str)),
        'calories': 'sum', 'protein_g': 'sum', 'fat_g': 'sum', 'carbs_g': 'sum'
    }).rename(columns={meal_col: 'Meal_Eaten', 'calories': 'Meal_Cals', 'protein_g': 'Meal_P', 'fat_g': 'Meal_F', 'carbs_g': 'Meal_C'})

    master_df = master_df.join(df_meals_agg)
    master_df['Is_Meal_Event'] = master_df['Meal_Eaten'].notna().astype(int)
except Exception as e:
    print(f"⚠️ Error fusing meal data: {e}")
    master_df['Is_Meal_Event'] = 0

# --- 7. EXPORT ---
master_df.to_csv('v15_master_telemetry.csv')
print(f"✅ FUSION COMPLETE. v15_master_telemetry.csv generated with {len(master_df)} 15-minute epochs.")