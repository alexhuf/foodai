import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import signal

print("🌌 V8 DREAMER: INITIALIZING ANALYTIC TEMPORAL EXPLORATION...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. GRACEFUL EXIT ---
class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        print("\n\n⚠️ [INTERRUPT DETECTED] Graceful Shutdown Initiated...")
        self.kill_now = True

killer = GracefulKiller()

# --- 2. LOAD THE UNIVERSE ---
with open('v8_physics_env.pkl', 'rb') as f:
    env = pickle.load(f)

class TemporalWorldModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.physics_engine = nn.Sequential(
            nn.Linear(state_dim, 512), nn.LayerNorm(512), nn.Mish(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.Mish(),
            nn.Linear(512, state_dim)
        )
    def forward(self, state_t): return state_t + self.physics_engine(state_t)

world_model = TemporalWorldModel(state_dim=len(env['feature_cols'])).to(device)
world_model.load_state_dict(torch.load('v8_world_model_weights.pth', map_location=device))
# We explicitly DO NOT freeze the weights entirely, we just set to eval so gradients can pass THROUGH it
world_model.eval() 

# --- 3. THE CHEF ACTOR ---
class ChefActor(nn.Module):
    def __init__(self, state_dim, action_dim=4): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid() 
        )
        self.max_limits = torch.tensor([300.0, 300.0, 500.0, 24.0]).to(device)

    def forward(self, state): 
        return self.net(state) * self.max_limits

state_dim = len(env['feature_cols'])
actor = ChefActor(state_dim).to(device)
actor_opt = optim.AdamW(actor.parameters(), lr=5e-4)

# --- 4. CHECKPOINT RECOVERY ---
checkpoint_path = 'v8_dreamer_checkpoint.pth'
start_episode = 0

if os.path.exists(checkpoint_path):
    print("🔄 Existing Checkpoint Found! Restoring memory banks...")
    checkpoint = torch.load(checkpoint_path)
    actor.load_state_dict(checkpoint['actor_state'])
    actor_opt.load_state_dict(checkpoint['actor_opt'])
    start_episode = checkpoint['episode']
    print(f"▶️ Resuming simulation from Episode {start_episode}")

# --- 5. THE REWARD FUNCTION (FULLY DIFFERENTIABLE) ---
TARGET_P, TARGET_F, TARGET_C, TARGET_FAST = 99.0, 97.0, 217.0, 6.7

def calculate_reward(simulated_actions):
    # We remove .item() so the PyTorch gradients remain attached to the math
    avg_p = simulated_actions[:, 0].mean()
    avg_f = simulated_actions[:, 1].mean()
    avg_c = simulated_actions[:, 2].mean()
    avg_fast = simulated_actions[:, 3].mean()
    
    penalty = torch.abs(avg_p - TARGET_P) + torch.abs(avg_f - TARGET_F) + torch.abs(avg_c - TARGET_C) + (torch.abs(avg_fast - TARGET_FAST) * 10)
    reward = -penalty 
    
    volatility = torch.std(simulated_actions, dim=0).sum()
    reward -= (volatility * 0.5) 
    return reward

# --- 6. THE MULTI-DAY SIMULATION LOOP ---
print("🚀 IGNITING ANALYTIC EXPLORATION. Press Ctrl+C at any time to safely pause.")

current_state = torch.randn(1, state_dim).to(device)
episodes = 500000 

for episode in range(start_episode, episodes):
    if killer.kill_now:
        break
        
    actor_opt.zero_grad()
    
    simulated_actions = []
    state = current_state.clone()
    
    for day in range(5):
        action = actor(state) 
        simulated_actions.append(action)
        
        # Inject the action physically into the state vector, keeping gradients intact
        state_modified = torch.cat([action, state[:, 4:]], dim=1)
        
        # The World Model predicts tomorrow. We removed no_grad(), so the Chef sees the consequences!
        next_state = world_model(state_modified)
        state = next_state
        
    sim_actions_tensor = torch.cat(simulated_actions, dim=0)
    timeline_reward = calculate_reward(sim_actions_tensor)
    
    # We want to maximize the reward. Optimizers minimize loss. So Loss = -Reward.
    loss = -timeline_reward 
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor_opt.step()
    
    # Randomly mutate the starting biology to force the AI to handle different scenarios
    current_state = current_state + (torch.randn(1, state_dim).to(device) * 0.1)

    if episode % 1000 == 0:
        p_val = sim_actions_tensor[:,0].mean().item()
        f_val = sim_actions_tensor[:,1].mean().item()
        c_val = sim_actions_tensor[:,2].mean().item()
        print(f"Simulation {episode:06d} | 5-Day Reward Score: {timeline_reward.item():.2f} | Avg Payload: P:{p_val:.0f}g F:{f_val:.0f}g C:{c_val:.0f}g")

# --- 7. THE FREEZE PROTOCOL ---
print("\n💾 Freezing Neural Weights to Disk...")
torch.save({
    'episode': episode,
    'actor_state': actor.state_dict(),
    'actor_opt': actor_opt.state_dict(),
}, checkpoint_path)

print("✅ CHECKPOINT SAVED. The GPU has been released.")