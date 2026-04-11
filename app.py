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

st.set_page_config(page_title="V8: Latent Space Navigator", page_icon="🧬", layout="wide")

# --- 1. ARCHITECTURE DEFINITIONS ---
class ChefActor(nn.Module):
    def __init__(self, state_dim, action_dim=4): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid() 
        )
        self.max_limits = torch.tensor([300.0, 300.0, 500.0, 24.0])
    def forward(self, state): return self.net(state) * self.max_limits

# --- 2. THE CONTEXT ENGINE (Smart Defaults) ---
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

# --- 3. LOAD THE UNIVERSE & COMPRESS TO LATENT SPACE ---
@st.cache_resource
def initialize_latent_space():
    with open('v8_physics_env.pkl', 'rb') as f:
        env = pickle.load(f)
    
    state_dim = len(env['feature_cols'])
    
    # Load AI Chef
    actor = ChefActor(state_dim)
    actor.load_state_dict(torch.load('v8_dreamer_checkpoint.pth', map_location='cpu')['actor_state'])
    actor.eval()
    
    # Load Matrix
    df = pd.read_csv('v5_omni_matrix.csv')
    baseline = df.iloc[-1].copy()
    
    # Find Meal Column robustly
    possible_names = ['Display_Name', 'Name', 'Meal', 'Food', 'Sequence_Target_String']
    meal_col = next((c for c in possible_names if c in df.columns), df.select_dtypes(include=['object']).columns[0])
    
    # --- BUG FIX: ROBUST TIME PARSING ---
    # Strip the timezone tail (-04:00, -08:00, Z) but let Pandas properly read the AM/PM format
    clean_time = df['Exact_Time'].astype(str).str.replace(r'([-+]\d{2}:?\d{2}|Z)$', '', regex=True)
    df['hour'] = pd.to_datetime(clean_time, errors='coerce').dt.hour
    
    def assign_slot(h):
        if pd.isna(h): return "Unknown"
        if 5 <= h < 11: return "Breakfast"
        elif 11 <= h < 15: return "Lunch"
        elif 15 <= h < 22: return "Dinner"
        else: return "Late Snack"
    df['Meal_Slot'] = df['hour'].apply(assign_slot)
    
    # Create the Historical Database
    meal_db = df.groupby([meal_col, 'Meal_Slot']).agg({
        'protein_g': 'mean', 'fat_g': 'mean', 'carbs_g': 'mean', 'calories': 'mean'
    }).reset_index().rename(columns={meal_col: 'Meal_Name'})
    
    # Run PCA to compress P/F/C into a 2D visual map
    pca = PCA(n_components=2)
    features = meal_db[['protein_g', 'fat_g', 'carbs_g']].fillna(0)
    latent_coords = pca.fit_transform(features)
    meal_db['Latent_X'] = latent_coords[:, 0]
    meal_db['Latent_Y'] = latent_coords[:, 1]
    
    return actor, env, baseline, meal_db, pca

actor, env, baseline, meal_db, pca = initialize_latent_space()

# Determine current likely slot
current_hour = current_time.hour
if 5 <= current_hour < 11: default_slot = "Breakfast"
elif 11 <= current_hour < 15: default_slot = "Lunch"
elif 15 <= current_hour < 22: default_slot = "Dinner"
else: default_slot = "Late Snack"

# --- 4. THE NAVIGATOR UI ---
st.title("🧬 V8 Latent Space Navigator")
st.markdown("Your biological history compressed into a geometric plane. Adjust your current state parameters to drag the Target Vector and find the optimal meal cluster.")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("⚙️ Vector Overrides")
    st.caption(f"Anchored to Okemos, MI | {current_time.strftime('%A, %I:%M %p')} | {current_temp:.1f}°F")
    
    # Time Filters
    st.markdown("**Temporal Filter**")
    selected_slot = st.selectbox("Isolate the map to this specific meal window:", 
                                 ["All", "Breakfast", "Lunch", "Dinner", "Late Snack"], 
                                 index=["All", "Breakfast", "Lunch", "Dinner", "Late Snack"].index(default_slot),
                                 help="Filters the geometric map to only show meals you have historically eaten during this time window.")
    
    st.divider()
    st.subheader("Biological State")
    
    sim_fasting = st.slider(
        "Fasting Window (Hours)", 
        min_value=0.0, max_value=24.0, value=6.0, step=0.5,
        help="How long since your last meal. Higher values push the AI's target vector toward dense, calorically heavy recovery foods to stabilize your energy flux."
    )
    
    sim_guilt = st.slider(
        "Guilt Delta (Yesterday's Deficit/Surplus)", 
        min_value=-2000, max_value=2000, value=0, step=50,
        help="Yesterday's caloric balance. Positive numbers (you overate) steer the AI toward lighter, fibrous, low-calorie nodes. Negative numbers (you starved) push it toward comfort and replenishment."
    )
    
    sim_temp = st.slider(
        "Ambient Temp Overlay (°F)", 
        min_value=0.0, max_value=100.0, value=float(current_temp), step=1.0,
        help="Simulated outside weather. Colder temperatures naturally shift the latent target toward warm, fat-heavy comfort clusters."
    )
    
    st.divider()
    st.subheader("Psychological State")
    
    sim_friction = st.slider(
        "Preparation Friction Tolerance", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="Your willingness to cook right now. 0.0 means 'I need delivery immediately' (pulls toward fast food clusters). 1.0 means 'I am ready to bake from scratch' (pulls toward high-effort, home-cooked clusters)."
    )
    
    sim_volatility = st.slider(
        "Noise Injection (Creativity)", 
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Injects mathematical chaos into the AI's prediction. 0.0 forces the AI to stick strictly to your predictable habits. Higher values force the Oracle to explore weird, forgotten, highly novel corners of your map."
    )

with col_right:
    # 1. Update State Vector
    input_dict = baseline[env['feature_cols']].to_dict()
    input_dict['Fasting_Hours_Current'] = sim_fasting
    input_dict['Guilt_Delta_24H'] = sim_guilt
    input_dict['Approx_High_Temp'] = sim_temp
    
    input_vector = pd.DataFrame([input_dict])[env['feature_cols']].fillna(0)
    scaled_vector = (input_vector.values - env['mean_state']) / env['std_state']
    state_tensor = torch.FloatTensor(scaled_vector)
    
    # 2. Ask the Dreamer what the target payload is
    with torch.no_grad():
        action = actor(state_tensor)
        
    # Inject noise based on the creativity slider
    noise = (torch.randn_like(action) * sim_volatility * 10).numpy()[0]
    target_p, target_f, target_c, _ = np.clip(action[0].numpy() + noise, a_min=0, a_max=None)
    
    # 3. Transform the Target Payload into the PCA Latent Space
    target_latent = pca.transform([[target_p, target_f, target_c]])[0]
    
    # 4. Filter the Database
    display_db = meal_db.copy()
    if selected_slot != "All":
        display_db = display_db[display_db['Meal_Slot'] == selected_slot]
    
    # Calculate Geometric Distance to Target
    display_db['Target_Distance'] = np.sqrt(
        (display_db['Latent_X'] - target_latent[0])**2 + 
        (display_db['Latent_Y'] - target_latent[1])**2
    )
    
    top_matches = display_db.sort_values('Target_Distance').head(3)

    # 5. Build the Interactive Plotly Graph
    fig = px.scatter(display_db, x='Latent_X', y='Latent_Y', 
                     color='Meal_Slot', hover_name='Meal_Name',
                     hover_data={'protein_g':':.0f', 'fat_g':':.0f', 'carbs_g':':.0f', 'Latent_X':False, 'Latent_Y':False},
                     title=f"Biological Latent Space ({selected_slot})",
                     template="plotly_dark", size_max=10)
    
    # Plot the "Target Pin"
    fig.add_trace(go.Scatter(
        x=[target_latent[0]], y=[target_latent[1]],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='cross'),
        name='Current Vector',
        text=["🎯 TARGET"], textposition="top right"
    ))
    
    # Draw Relationship Lines to the 3 closest meals
    for _, match in top_matches.iterrows():
        fig.add_trace(go.Scatter(
            x=[target_latent[0], match['Latent_X']],
            y=[target_latent[1], match['Latent_Y']],
            mode='lines',
            line=dict(color='white', width=1, dash='dash'),
            showlegend=False
        ))

    st.plotly_chart(fig, use_container_width=True)

    # 6. Output the Data
    st.subheader("🎯 Mathematical Direct Hits")
    st.markdown(f"**Optimal Target Payload:** {target_p:.0f}g P | {target_f:.0f}g F | {target_c:.0f}g C")
    
    if not top_matches.empty:
        cols = st.columns(3)
        for i, (idx, match) in enumerate(top_matches.iterrows()):
            with cols[i]:
                st.success(f"**{match['Meal_Name']}**")
                st.caption(f"Geodesic Distance: {match['Target_Distance']:.1f}")
                st.write(f"P: {match['protein_g']:.0f}g | F: {match['fat_g']:.0f}g | C: {match['carbs_g']:.0f}g")
    else:
        st.warning("No historical meals found in this time slot.")