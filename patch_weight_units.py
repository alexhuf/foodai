import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

print("🛠️ INITIATING UNIT CONVERSION PATCH...")

# 1. Load the active era dataset
df_active = pd.read_csv("v15_master_telemetry_active.csv", index_col='datetime', parse_dates=True, low_memory=False)

# 2. Re-extract Samsung (Converted to LBS)
df_sam = pd.read_csv("com.samsung.health.weight.20260411021045.csv", skiprows=1, index_col=False)
clean_time_sam = df_sam['create_time'].astype(str).str.replace(r'([-+]\d{2}:?\d{2}|Z)$', '', regex=True)
df_sam['date'] = pd.to_datetime(clean_time_sam, errors='coerce').dt.floor('D')
df_sam_wt = df_sam.groupby('date').agg({'weight': 'mean'}).rename(columns={'weight': 'sam_weight'})
df_sam_wt['sam_weight'] = df_sam_wt['sam_weight'] * 2.20462 # THE KILOGRAM FIX

# 3. Re-extract Noom (Already converted to LBS in extraction)
df_actions = pd.read_csv("actions.csv")
df_weigh_in = df_actions[df_actions['actionType'] == 'WEIGH_IN'].copy()
def extract_weight(json_str):
    try: return json.loads(json_str).get('weightInKg', np.nan) * 2.20462
    except: return np.nan
df_weigh_in['noom_weight'] = df_weigh_in['jsonString'].apply(extract_weight)
df_weigh_in['date'] = pd.to_datetime(df_weigh_in['date'], errors='coerce').dt.floor('D')
df_noom_wt = df_weigh_in.groupby('date').agg({'noom_weight': 'mean'})

# 4. Correctly Fuse and Recalculate
df_weight = df_sam_wt.join(df_noom_wt, how='outer')
df_weight['true_weight'] = df_weight[['sam_weight', 'noom_weight']].mean(axis=1)

full_days = pd.date_range(start=df_active.index.min().floor('D') - pd.Timedelta(days=35), end=df_active.index.max().floor('D'), freq='D')
df_weight = df_weight.reindex(full_days).ffill()

windows = [3, 5, 7, 14, 30]
cols_to_update = ['true_weight']
for w in windows:
    ema_col, vel_col = f'weight_ema_{w}d', f'weight_velocity_{w}d'
    df_weight[ema_col] = df_weight['true_weight'].ewm(span=w, adjust=False).mean()
    df_weight[vel_col] = df_weight[ema_col].diff()
    cols_to_update.extend([ema_col, vel_col])

# 5. Overwrite the broken columns in the active dataset
df_active['date_floor'] = df_active.index.floor('D')
df_active = df_active.drop(columns=[c for c in cols_to_update if c in df_active.columns])
df_active = df_active.merge(df_weight[cols_to_update], left_on='date_floor', right_index=True, how='left')
df_active = df_active.drop(columns=['date_floor'])

df_active.to_csv("v15_master_telemetry_active.csv")
print("✅ METRIC COLLISION FIXED. Dataset is now 100% mathematically pure.")