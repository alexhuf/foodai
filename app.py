import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import math

st.set_page_config(page_title="FoodAI V5: The Oracle", page_icon="🔮", layout="wide")

# --- 1. ARCHITECTURE DEFINITIONS (Must match exactly) ---
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

# --- 2. LOAD THE FROZEN BRAIN ---
@st.cache_resource
def load_oracle():
    try:
        with open('v5_env_objects.pkl', 'rb') as f:
            env = pickle.load(f)
        
        # Load the latest 120-day biological state from the Omni-Matrix
        df = pd.read_csv('v5_omni_matrix.csv')
        latest_baseline = df.iloc[-1].copy()
        
        # Initialize Model on CPU
        model = AutoregressivePlateGenerator(
            context_dim=len(env['feature_cols']), 
            vocab_size=len(env['vocab'])
        )
        
        # Load the weights
        model.load_state_dict(torch.load('v5_transformer_weights.pth', map_location=torch.device('cpu')))
        model.eval()
        
        return model, env, latest_baseline
    except Exception as e:
        st.error(f"Failed to load Oracle files. Ensure training completed successfully. Error: {e}")
        return None, None, None

model, env, baseline = load_oracle()

if model is None:
    st.stop()

# --- 3. THE DASHBOARD ---
st.title("🔮 FoodAI V5: The Autoregressive Oracle")
st.markdown("The 5070 Ti has mathematically mapped your subconscious. Inject volatile triggers into your current baseline to see what your body generates.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Volatile Psychological Triggers")
    st.write("Alter these to simulate different conditions today:")
    
    # The most heavily weighted psychological triggers
    sim_fasting = st.slider("Hours Fasting", 0.0, 24.0, float(baseline.get('Fasting_Hours_Current', 4.0)))
    sim_guilt = st.slider("Yesterday's Caloric Deficit (Guilt Delta)", -2000, 2000, int(baseline.get('Guilt_Delta_24H', 0)))
    sim_budget = st.slider("Remaining Calories Today", -500, 3000, int(baseline.get('Remaining_Budget', 1200)))
    sim_steps = st.slider("Steps Taken So Far", 0, 25000, int(baseline.get('steps', 5000)))
    sim_temp = st.slider("Outside Temperature (°F)", 0, 100, int(baseline.get('Approx_High_Temp', 65)))
    
    st.divider()
    st.subheader("Inference Settings")
    sim_creativity = st.slider("Neural Creativity (Temperature)", 0.1, 2.5, 1.2, 0.1)
    st.caption("Lower = Strict Habits. Higher = Exploring biological edge cases.")

with col2:
    st.header("The Reconstitution Engine")
    
    if st.button("🧬 GENERATE PLATE", type="primary", use_container_width=True):
        
        # 1. Update the baseline vector with the simulation data
        input_dict = baseline[env['feature_cols']].to_dict()
        input_dict['Fasting_Hours_Current'] = sim_fasting
        input_dict['Guilt_Delta_24H'] = sim_guilt
        input_dict['Remaining_Budget'] = sim_budget
        input_dict['steps'] = sim_steps
        input_dict['Approx_High_Temp'] = sim_temp
        
        # 2. Convert and Scale
        input_vector = pd.DataFrame([input_dict])[env['feature_cols']].fillna(0)
        scaled_vector = env['scaler'].transform(input_vector)
        context_tensor = torch.FloatTensor(scaled_vector)
        
        # 3. AUTOREGRESSIVE DECODING (With Multinomial Sampling)
        with torch.no_grad():
            memory = model.context_projector(context_tensor).unsqueeze(1)
            
            target_seq = torch.tensor([[env['vocab']['<SOS>']]])
            generated_plate = []
            
            for _ in range(env['max_seq_len']):
                tgt_emb = model.pos_encoder(model.food_embedding(target_seq))
                tgt_mask = model.generate_square_subsequent_mask(target_seq.size(1))
                
                out = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # Apply Temperature Scaling to the logits
                logits = model.output_layer(out[:, -1, :]) / sim_creativity
                
                # Convert logits to a strict probability distribution
                probs = torch.softmax(logits, dim=-1)
                
                # Roll a weighted mathematical dice based on those probabilities
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == env['vocab']['<EOS>']:
                    break
                    
                ingredient_name = env['idx_to_vocab'][next_token]
                generated_plate.append(ingredient_name)
                
                target_seq = torch.cat([target_seq, torch.tensor([[next_token]])], dim=1)
        
        # 4. Display the Generated Sequence
        st.success("### Mathematically Generated Plate:")
        if not generated_plate:
             st.warning("The Oracle generated an empty sequence. Try adjusting the parameters or creativity.")
        else:
            for idx, food in enumerate(generated_plate):
                st.markdown(f"#### {idx + 1}. **{food}**")
            
        st.caption(f"Sequence generated chronologically with a creativity temperature of {sim_creativity}. The AI chose the main ingredient first, then selected the sides conditionally based on both your biology and the preceding dish.")