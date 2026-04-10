import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import math
import requests
from datetime import datetime

st.set_page_config(page_title="FoodAI V6: Analytical Oracle", page_icon="🧬", layout="wide")

# --- 1. ARCHITECTURE DEFINITIONS ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class AutoregressivePlateGenerator(nn.Module):
    def __init__(self, context_dim, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.food_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- 2. LIVE CONTEXT ENGINE ---
@st.cache_data(ttl=3600) # Cache for 1 hour to avoid API spam
def get_live_context():
    # Okemos, MI Coordinates
    lat, lon = 42.723, -84.400
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,is_day&temperature_unit=fahrenheit"
        response = requests.get(url).json()
        temp = response['current']['temperature_2m']
    except:
        temp = 60.0 # Fallback
        
    now = datetime.now()
    return temp, now

# --- 3. LOAD FROZEN BRAIN ---
@st.cache_resource
def load_oracle():
    with open('v5_env_objects.pkl', 'rb') as f:
        env = pickle.load(f)
    
    df = pd.read_csv('v5_omni_matrix.csv')
    latest_baseline = df.iloc[-1].copy()
    
    # Calculate historical averages for the specific sliders
    historical_avgs = {
        'Fasting': df['Fasting_Hours_Current'].mean(),
        'Budget': df['Remaining_Budget'].mean(),
        'Steps': df['steps'].mean()
    }
    
    model = AutoregressivePlateGenerator(context_dim=len(env['feature_cols']), vocab_size=len(env['vocab']))
    model.load_state_dict(torch.load('v5_transformer_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model, env, latest_baseline, historical_avgs

model, env, baseline, hist_avgs = load_oracle()
live_temp, current_time = get_live_context()

# --- 4. DASHBOARD UI ---
st.title("🧬 FoodAI V6: The Analytical Oracle")
st.markdown("Live Environmental Sync Active. The AI is analyzing multiple divergent paths based on your current physiological state.")

st.sidebar.header("🌍 Live Environment")
st.sidebar.metric("Current Local Time", current_time.strftime("%I:%M %p"))
st.sidebar.metric("Live Temp (Okemos, MI)", f"{live_temp:.1f} °F")
st.sidebar.metric("Day of Week", current_time.strftime("%A"))

st.sidebar.divider()
st.sidebar.header("⚖️ Biological State Overrides")
st.sidebar.caption("Sliders default to your historical rolling averages. Alter them to test hypotheses.")

sim_fasting = st.sidebar.slider("Hours Fasting", 0.0, 24.0, float(hist_avgs['Fasting']))
sim_guilt = st.sidebar.slider("Yesterday's Caloric Deficit (Guilt Delta)", -2000, 2000, int(baseline.get('Guilt_Delta_24H', 0)))
sim_budget = st.sidebar.slider("Remaining Calories Today", -500, 3000, int(hist_avgs['Budget']))
sim_steps = st.sidebar.slider("Steps Taken So Far", 0, 25000, int(hist_avgs['Steps']))
sim_temp = st.sidebar.slider("Override Temperature (°F)", 0, 100, int(live_temp))

# --- 5. THE ANALYTICAL GENERATOR ---
if st.button("🧠 EXECUTE DEEP ANALYSIS", type="primary", use_container_width=True):
    
    # 1. Update State Vector
    input_dict = baseline[env['feature_cols']].to_dict()
    input_dict['Fasting_Hours_Current'] = sim_fasting
    input_dict['Guilt_Delta_24H'] = sim_guilt
    input_dict['Remaining_Budget'] = sim_budget
    input_dict['steps'] = sim_steps
    input_dict['Approx_High_Temp'] = sim_temp
    
    input_vector = pd.DataFrame([input_dict])[env['feature_cols']].fillna(0)
    scaled_vector = env['scaler'].transform(input_vector)
    context_tensor = torch.FloatTensor(scaled_vector)
    
    # 2. Extract Top Primary Cravings (Explainability)
    with torch.no_grad():
        memory = model.context_projector(context_tensor).unsqueeze(1)
        target_seq = torch.tensor([[env['vocab']['<SOS>']]])
        
        # Analyze the very first decision (The Main Dish)
        tgt_emb = model.pos_encoder(model.food_embedding(target_seq))
        tgt_mask = model.generate_square_subsequent_mask(1)
        out = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        raw_logits = model.output_layer(out[:, -1, :])
        probs = torch.softmax(raw_logits, dim=-1)[0]
        
        # Get Top 3 Main Dishes
        top_probs, top_indices = torch.topk(probs, 3)
    
    st.divider()
    st.subheader("📊 Biological Push Analysis (The 'Why')")
    st.write(f"Based on your {sim_fasting:.1f} hour fast, the {sim_temp}°F weather, and your current energy flux, your neural network is demanding these core anchors:")
    
    col_a, col_b, col_c = st.columns(3)
    metrics = [col_a, col_b, col_c]
    
    for i in range(3):
        food_name = env['idx_to_vocab'][top_indices[i].item()]
        probability = top_probs[i].item() * 100
        metrics[i].metric(label=f"Anchor {i+1}", value=food_name, delta=f"{probability:.1f}% Probability", delta_color="normal")
    
    st.divider()
    st.subheader("🌌 Divergent Mathematical Pathways")
    st.write("The Oracle has hallucinated three distinct meals anchored around those primary probabilities.")
    
    col1, col2, col3 = st.columns(3)
    
    # Generate 3 distinct paths using the top 3 anchors to force diversity
    for i, column in enumerate([col1, col2, col3]):
        with column:
            with torch.no_grad():
                anchor_token = top_indices[i].item()
                current_seq = torch.tensor([[env['vocab']['<SOS>'], anchor_token]])
                generated_plate = [env['idx_to_vocab'][anchor_token]]
                
                # Autoregressively finish the plate conditionally based on the anchor
                for _ in range(env['max_seq_len'] - 1):
                    tgt_emb = model.pos_encoder(model.food_embedding(current_seq))
                    tgt_mask = model.generate_square_subsequent_mask(current_seq.size(1))
                    out = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    
                    # Moderate temperature to keep it realistic but distinct
                    logits = model.output_layer(out[:, -1, :]) / 1.1 
                    next_token = torch.argmax(logits, dim=-1).item() # Greedy finish for coherence
                    
                    if next_token == env['vocab']['<EOS>']:
                        break
                        
                    generated_plate.append(env['idx_to_vocab'][next_token])
                    current_seq = torch.cat([current_seq, torch.tensor([[next_token]])], dim=1)
                
                st.success(f"**Path {i+1}: {['Primary', 'Secondary', 'Tertiary'][i]} Craving**")
                for idx, food in enumerate(generated_plate):
                    st.markdown(f"- {food}")