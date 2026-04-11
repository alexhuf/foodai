import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from datetime import datetime
import requests

st.set_page_config(page_title="V9: Live MPC Navigator", page_icon="🧬", layout="wide")

# --- 1. ARCHITECTURE: THE BIOLOGICAL PHYSICS ENGINE ---
class TemporalWorldModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.physics_engine = nn.Sequential(
            nn.Linear(state_dim, 512), nn.LayerNorm(512), nn.Mish(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.Mish(),
            nn.Linear(512, state_dim)
        )
    def forward(self, state_t): return state_t + self.physics_engine(state_t)

# --- 2. THE CONTEXT ENGINE ---
@st.cache_data(ttl=3600)
def get_live_context():
    lat, lon = 42.723, -84.400 # Okemos, MI
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m&temperature_unit=fahrenheit"
        response = requests.get(url).json()
        temp = response['current']['temperature_2m']
    except:
        temp = 45.0 
    return temp, datetime.now()

current_temp, current_time = get_live_context()

# --- 3. LOAD THE UNIVERSE ---
@st.cache_resource
def initialize_latent_space():
    with open('v8_physics_env.pkl', 'rb') as f:
        env = pickle.load(f)
    
    state_dim = len(env['feature_cols'])
    
    # Load the Simulator (We threw away the ChefActor)
    world_model = TemporalWorldModel(state_dim)
    world_model.load_state_dict(torch.load('v8_world_model_weights.pth', map_location='cpu'))
    world_model.eval()
    
    df = pd.read_csv('v5_omni_matrix.csv')
    baseline = df.iloc[-1].copy()
    
    possible_names = ['Display_Name', 'Name', 'Meal', 'Food', 'Sequence_Target_String']
    meal_col = next((c for c in possible_names if c in df.columns), df.select_dtypes(include=['object']).columns[0])
    
    clean_time = df['Exact_Time'].astype(str).str.replace(r'([-+]\d{2}:?\d{2}|Z)$', '', regex=True)
    df['hour'] = pd.to_datetime(clean_time, errors='coerce').dt.hour
    
    def assign_slot(h):
        if pd.isna(h): return "Unknown"
        if 5 <= h < 11: return "Breakfast"
        elif 11 <= h < 15: return "Lunch"
        elif 15 <= h < 22: return "Dinner"
        else: return "Late Snack"
    df['Meal_Slot'] = df['hour'].apply(assign_slot)
    
    meal_db = df.groupby([meal_col, 'Meal_Slot']).agg({
        'protein_g': 'mean', 'fat_g': 'mean', 'carbs_g': 'mean', 'calories': 'mean'
    }).reset_index().rename(columns={meal_col: 'Meal_Name'})
    
    pca = PCA(n_components=2)
    features = meal_db[['protein_g', 'fat_g', 'carbs_g']].fillna(0)
    latent_coords = pca.fit_transform(features)
    meal_db['Latent_X'] = latent_coords[:, 0]
    meal_db['Latent_Y'] = latent_coords[:, 1]
    
    return world_model, env, baseline, meal_db, pca

world_model, env, baseline, meal_db, pca = initialize_latent_space()

# Temporal Slot Multipliers (Prevents the "Impossible Meal")
slot_ratios = {
    "Breakfast": 0.25, # 25% of daily macros
    "Lunch": 0.30,
    "Dinner": 0.35,
    "Late Snack": 0.10,
    "All": 0.33
}

current_hour = current_time.hour
if 5 <= current_hour < 11: default_slot = "Breakfast"
elif 11 <= current_hour < 15: default_slot = "Lunch"
elif 15 <= current_hour < 22: default_slot = "Dinner"
else: default_slot = "Late Snack"

# --- 4. THE NAVIGATOR UI ---
st.title("🧬 V9 Latent Space Navigator: MPC")
st.markdown("Real-time Model Predictive Control. Sliders instantly alter the biological simulation, forcing the Target Vector to hunt for state-stabilizing payloads.")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("⚙️ Vector Overrides")
    st.caption(f"Anchored to Okemos, MI | {current_temp:.1f}°F")
    
    selected_slot = st.selectbox("Temporal Filter (Scales Payload)", ["All", "Breakfast", "Lunch", "Dinner", "Late Snack"], index=["All", "Breakfast", "Lunch", "Dinner", "Late Snack"].index(default_slot))
    
    st.divider()
    sim_fasting = st.slider("Fasting Window (Hours)", 0.0, 24.0, 6.0, 0.5)
    sim_guilt = st.slider("Guilt Delta (Yesterday's Deficit/Surplus)", -2000, 2000, 0, 50)
    sim_temp = st.slider("Ambient Temp Overlay (°F)", 0.0, 100.0, float(current_temp), 1.0)
    
    # Target Fingerprint Control (Exposed to the user now)
    st.divider()
    st.subheader("Target Homeostasis")
    st.caption("The metabolic center the AI is trying to return your body to tomorrow.")
    target_p = st.number_input("Target Daily Protein", value=99)
    target_f = st.number_input("Target Daily Fat", value=97)
    target_c = st.number_input("Target Daily Carbs", value=217)

with col_right:
    # 1. Update Current State Vector
    input_dict = baseline[env['feature_cols']].to_dict()
    input_dict['Fasting_Hours_Current'] = sim_fasting
    input_dict['Guilt_Delta_24H'] = sim_guilt
    input_dict['Approx_High_Temp'] = sim_temp
    
    input_vector = pd.DataFrame([input_dict])[env['feature_cols']].fillna(0)
    scaled_vector = (input_vector.values - env['mean_state']) / env['std_state']
    current_state_tensor = torch.FloatTensor(scaled_vector)
    
    # --- V9 MODEL PREDICTIVE CONTROL (The Real-Time Brain) ---
    # Generate 2000 random meal payloads, scaled roughly to the time of day
    ratio = slot_ratios[selected_slot]
    base_p, base_f, base_c = target_p * ratio, target_f * ratio, target_c * ratio
    
    # Create a tensor of 2000 guesses [P, F, C, Fasting]
    guesses = torch.randn(2000, 4) 
    guesses[:, 0] = torch.clamp((guesses[:, 0] * 20) + base_p, min=0, max=150) # P
    guesses[:, 1] = torch.clamp((guesses[:, 1] * 20) + base_f, min=0, max=150) # F
    guesses[:, 2] = torch.clamp((guesses[:, 2] * 40) + base_c, min=0, max=250) # C
    guesses[:, 3] = sim_fasting # Lock fasting to the slider
    
    # Broadcast the current state 2000 times to match the guesses
    batch_states = current_state_tensor.repeat(2000, 1)
    batch_states[:, :4] = guesses # Inject the 2000 different meals into the states
    
    # Run the 2000 meals through the physics simulator to see tomorrow
    with torch.no_grad():
        tomorrow_states = world_model(batch_states)
    
    # Grade the results: Which meal brought tomorrow's state closest to homeostasis?
    # (Assuming first 3 dims are the macros of tomorrow's state)
    penalty_p = torch.abs(tomorrow_states[:, 0] - ((target_p - env['mean_state'][0]) / env['std_state'][0]))
    penalty_f = torch.abs(tomorrow_states[:, 1] - ((target_f - env['mean_state'][1]) / env['std_state'][1]))
    penalty_c = torch.abs(tomorrow_states[:, 2] - ((target_c - env['mean_state'][2]) / env['std_state'][2]))
    
    # Add a penalty for extreme volatility in the rest of the 157 dimensions
    volatility = torch.std(tomorrow_states, dim=1)
    
    total_penalty = penalty_p + penalty_f + penalty_c + (volatility * 0.5)
    
    # The winner is the index with the lowest penalty
    best_idx = torch.argmin(total_penalty)
    best_action = guesses[best_idx].numpy()
    
    opt_p, opt_f, opt_c = best_action[0], best_action[1], best_action[2]
    
    # --- RENDER THE MAP ---
    target_latent = pca.transform([[opt_p, opt_f, opt_c]])[0]
    
    display_db = meal_db.copy()
    if selected_slot != "All":
        display_db = display_db[display_db['Meal_Slot'] == selected_slot]
    
    display_db['Target_Distance'] = np.sqrt((display_db['Latent_X'] - target_latent[0])**2 + (display_db['Latent_Y'] - target_latent[1])**2)
    top_matches = display_db.sort_values('Target_Distance').head(3)

    fig = px.scatter(display_db, x='Latent_X', y='Latent_Y', 
                     color='Meal_Slot', hover_name='Meal_Name',
                     hover_data={'protein_g':':.0f', 'fat_g':':.0f', 'carbs_g':':.0f', 'Latent_X':False, 'Latent_Y':False},
                     title=f"Biological Latent Space ({selected_slot})",
                     template="plotly_dark", size_max=10)
    
    fig.add_trace(go.Scatter(
        x=[target_latent[0]], y=[target_latent[1]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='cross'),
        name='Current Vector',
        text=["🎯 TARGET"], textposition="top right"
    ))
    
    for _, match in top_matches.iterrows():
        fig.add_trace(go.Scatter(
            x=[target_latent[0], match['Latent_X']],
            y=[target_latent[1], match['Latent_Y']],
            mode='lines',
            line=dict(color='white', width=1, dash='dash'),
            showlegend=False
        ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Real-Time Computed Target")
    st.markdown(f"**Optimal Single-Meal Payload:** {opt_p:.0f}g P | {opt_f:.0f}g F | {opt_c:.0f}g C")
    
    if not top_matches.empty:
        cols = st.columns(3)
        for i, (idx, match) in enumerate(top_matches.iterrows()):
            with cols[i]:
                st.success(f"**{match['Meal_Name']}**")
                st.caption(f"Geodesic Distance: {match['Target_Distance']:.1f}")
                st.write(f"P: {match['protein_g']:.0f}g | F: {match['fat_g']:.0f}g | C: {match['carbs_g']:.0f}g")