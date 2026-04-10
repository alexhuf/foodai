import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="FoodAI V3: Generative Oracle", page_icon="🧬", layout="centered")
st.title("🧬 FoodAI V3: The Generative Plate Oracle")
st.write("The AI has analyzed your 45-dimensional physiological and environmental state. Adjust your immediate context below to generate a plate.")

# --- 2. LOAD THE MEMORY ENGINE ---
@st.cache_resource
def load_brain():
    try:
        with open("v3_memory_engine.pkl", "rb") as f:
            brain = pickle.load(f)
        
        # Load the raw dataset just to grab your "Current/Latest" rolling averages as a baseline
        latest_df = pd.read_csv('V3_Deep_Plate_Dataset.csv')
        baseline = latest_df.iloc[-1].copy() # Get the absolute latest logged day
        
        return brain, baseline
    except FileNotFoundError:
        return None, None

brain, baseline = load_brain()

if brain is None:
    st.warning("⚠️ Waiting for V3 Brain files. Run `python build_v3_brain.py` first.")
    st.stop()

# --- 3. UI DASHBOARD ---
st.sidebar.header("Immediate Volatile State")

# The user adjusts the "right now" variables
hours_fasting = st.sidebar.slider("Hours Since Last Meal", 0.0, 24.0, 4.0, 0.5)
budget_left = st.sidebar.slider("Calories Left in Intraday Budget", -500, 2500, 1000)
high_temp = st.sidebar.slider("Outside Temp (°F)", 0, 100, 50)
current_steps = st.sidebar.slider("Steps Taken So Far", 0, 30000, int(baseline['steps']))
willpower_streak = st.sidebar.slider("Current Deficit Streak (Days)", 0, 30, int(baseline['Deficit_Streak']))

# --- 4. ASSEMBLE THE 45-DIMENSIONAL VECTOR ---
if st.button("🔮 Generate My Plate", use_container_width=True):
    
    # Start with your real, absolute latest baseline for the deep rolling averages
    input_dict = baseline[brain['feature_cols']].to_dict()
    
    # Overwrite the baseline with your live slider inputs
    input_dict['Hours_Since_Last_Meal'] = hours_fasting
    input_dict['Remaining_Intraday_Budget'] = budget_left
    input_dict['Approx_High_Temp'] = high_temp
    input_dict['steps'] = current_steps
    input_dict['Deficit_Streak'] = willpower_streak
    
    # Handle the boolean weather flags
    input_dict['Is_Freezing'] = 1 if high_temp <= 32 else 0
    input_dict['Is_Hot'] = 1 if high_temp >= 75 else 0
    
    # Convert String variables to their encoded numbers using the saved encoders
    for col in ['Day_of_Week', 'Month']:
        # If the baseline string is somehow missing, default to the first class
        val = str(input_dict[col])
        try:
            input_dict[col] = brain['encoders'][col].transform([val])[0]
        except ValueError:
            input_dict[col] = 0

    # Convert the dictionary into the exact flat array the AI needs
    input_vector = pd.DataFrame([input_dict])[brain['feature_cols']]
    
    # Scale the input
    vector_scaled = brain['scaler'].transform(input_vector)
    
    # --- 5. THE HYPERSPACE QUERY ---
    # Ask the KNN to find the 3 days in your life where you felt EXACTLY like this
    distances, indices = brain['knn_model'].kneighbors(vector_scaled, n_neighbors=3)
    
    st.divider()
    st.header("🍽️ The Oracle's Generated Plates")
    st.write(f"Based on your 45-dimensional physiological state, here are the plates your body is mathematically demanding:")
    
    ref_data = brain['reference_data']
    
    # Display the results
    for i, idx in enumerate(indices[0]):
        plate = ref_data.iloc[idx]
        
        # We calculate "Match Confidence" inversely from the Euclidean distance. 
        # (Closer to 0 distance = closer to 100% match)
        match_score = max(0, 100 - (distances[0][i] * 10)) 
        
        with st.container():
            st.subheader(f"Option {i+1} ({match_score:.1f}% Match)")
            st.success(f"**{plate['Full_Meal_Plate']}**")
            
            # Show the deep data for the generated plate
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Calories", int(plate['calories']))
            col2.metric("Protein", f"{int(plate['protein_g'])}g")
            col3.metric("Carbs", f"{int(plate['carbs_g'])}g")
            col4.metric("Archetype", plate['Plate_Archetype'].split(",")[0])
            st.write("---")