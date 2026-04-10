import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import math
import os

print("🧬 FOODAI V5: INITIALIZING AUTOREGRESSIVE CONDITIONAL TRANSFORMER...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Hardware Uplink Confirmed: Computing on {device.type.upper()}")

# --- 1. LOAD THE OMNI-MATRIX ---
print("⚙️ Loading 157-Dimensional Omni-Matrix...")
df = pd.read_csv('v5_omni_matrix.csv')

# Separate the Target Sequences from the Biological Context
sequences = df['Sequence_Target_String'].apply(lambda x: [item.strip() for item in str(x).split('|')])

# Isolate the 157 input features (Drop text/date columns)
exclude_cols = ['date', 'timeSlot', 'Meal_Time', 'Display_Name', 'Exact_Time', 'Sequence_Target_String']
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.int32, float, int]]

X_raw = df[feature_cols].fillna(0).values

# Scale the Biological Context
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- 2. BUILD THE FOOD VOCABULARY ---
print("📚 Constructing the Generative Food Lexicon...")
# Special Tokens: Padding, Start of Sequence, End of Sequence
PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2
vocab = {'<PAD>': PAD_IDX, '<SOS>': SOS_IDX, '<EOS>': EOS_IDX}

idx = 3
for seq in sequences:
    for food in seq:
        if food not in vocab:
            vocab[food] = idx
            idx += 1

idx_to_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)
print(f"Total Unique Ingredients (Tokens) Mapped: {vocab_size - 3}")

# --- 3. SEQUENCE PADDING & TENSOR CONVERSION ---
max_seq_len = max(len(seq) for seq in sequences) + 2 # +2 for SOS and EOS

def encode_sequence(seq):
    encoded = [SOS_IDX] + [vocab[food] for food in seq] + [EOS_IDX]
    padding_length = max_seq_len - len(encoded)
    encoded.extend([PAD_IDX] * padding_length)
    return encoded

y_encoded = np.array([encode_sequence(seq) for seq in sequences])

class PlateDataset(Dataset):
    def __init__(self, context, sequences):
        self.context = torch.FloatTensor(context)
        self.sequences = torch.LongTensor(sequences)

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        # Input to decoder is sequence except last token
        # Target for decoder is sequence except first token (shifted by 1)
        return self.context[idx], self.sequences[idx, :-1], self.sequences[idx, 1:]

dataset = PlateDataset(X_scaled, y_encoded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 4. THE CONDITIONAL TRANSFORMER ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class AutoregressivePlateGenerator(nn.Module):
    def __init__(self, context_dim, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 1. Biological State Encoder (Compresses 157D into Transformer dimensions)
        self.context_projector = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Food Sequence Embedding
        self.food_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. The Core Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. Output Logic
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        # Prevents the AI from "looking ahead" at the answer while predicting the next ingredient
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, context, target_seq):
        # Context becomes the "Memory" for the Transformer
        memory = self.context_projector(context).unsqueeze(1) # (Batch, 1, d_model)
        
        # Embed and add positional encoding to the food sequence
        tgt_emb = self.pos_encoder(self.food_embedding(target_seq))
        
        # Create causal mask for autoregressive training
        tgt_mask = self.generate_square_subsequent_mask(target_seq.size(1)).to(target_seq.device)
        
        # The Transformer compares the Sequence against the Biological Memory
        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_layer(out)

# --- 5. THE GPU TRAINING CRUCIBLE ---
print("⚔️ Igniting the 5070 Ti Training Crucible...")

model = AutoregressivePlateGenerator(context_dim=X_scaled.shape[1], vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# We will run 150 deep epochs for this core architecture
epochs = 150
model.train()

for epoch in range(epochs):
    total_loss = 0
    for context, seq_in, seq_out in dataloader:
        context, seq_in, seq_out = context.to(device), seq_in.to(device), seq_out.to(device)
        
        optimizer.zero_grad()
        predictions = model(context, seq_in)
        
        # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize) vs (Batch * SeqLen)
        loss = criterion(predictions.reshape(-1, vocab_size), seq_out.reshape(-1))
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Transformer Loss: {total_loss/len(dataloader):.4f}")

# --- 6. FREEZE AND EXPORT ---
print("\n📦 Freezing Neural Weights...")
torch.save(model.state_dict(), 'v5_transformer_weights.pth')

with open('v5_env_objects.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'vocab': vocab,
        'idx_to_vocab': idx_to_vocab,
        'feature_cols': feature_cols,
        'max_seq_len': max_seq_len
    }, f)

print("✅ V5 TRANSFORMER TRAINING COMPLETE. The Oracle has been forged.")