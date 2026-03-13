import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import pickle

# 1. Define the Mini-GPT Transformer Architecture (Must match training)
class MiniGPT_IDS(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, window_size, features)
        x = nn.Dense(self.embed_dim)(x)
        
        # Self-Attention (The "Logic" layer)
        attn_out = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = x + attn_out 
        x = nn.LayerNorm()(x)
        
        # Feed Forward
        ff_out = nn.Dense(self.embed_dim * 2)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.embed_dim)(ff_out)
        x = x + ff_out
        
        # Predict the 5 features of the NEXT message
        return nn.Dense(5)(x[:, -1, :])

# 2. Load the trained Parameters and Test Data
print("Loading trained model and data...")
with open('ids_model_params.pkl', 'rb') as f:
    params = pickle.load(f)

X_test = jnp.load('X_train.npy')
model = MiniGPT_IDS(embed_dim=32, num_heads=4)

# 3. Define the Anomaly Scorer
def get_anomaly_score(sequence, actual_next):
    # Predict what SHOULD come next
    pred = model.apply({'params': params}, sequence.reshape(1, 10, 5))
    # Calculate Mean Squared Error (The "Surprise" factor)
    error = jnp.mean(jnp.square(pred - actual_next))
    return error

# --- SCENARIO 1: NORMAL DATA ---
# Pick a sequence from the log (Sequence #500)
seq_normal = X_test[500] 
actual_next = X_test[501][-1] # This is what actually followed in the log
score_normal = get_anomaly_score(seq_normal, actual_next)

# --- SCENARIO 2: ATTACK DATA (Spoofed Speed) ---
# We use the same sequence, but we INJECT a fake speed value.
# JAX Fix: use .at[index].set(value) because arrays are immutable.
# Index 1 corresponds to the Speed Byte (d0 of ID 0x102) in our vector.
next_attack = actual_next.at[1].set(0.95) # Simulating a 240km/h spike!

score_attack = get_anomaly_score(seq_normal, next_attack)

# 4. Output the Comparison
print("\n" + "="*40)
print("IDS PERFORMANCE EVALUATION")
print("="*40)
print(f"Normal Data Error: {score_normal:.8f}")
print(f"Attack Data Error: {score_attack:.8f}")
print("-" * 40)

# Detection Logic:
anomaly_ratio = score_attack / score_normal
print(f"Anomaly Ratio: {anomaly_ratio:.2f}x")

if anomaly_ratio > 10.0:
    print("\n[!!!] ALERT: INTRUSION DETECTED [!!!]")
    print("STATUS: Model 'Surprise' exceeded threshold.")
    print("REASON: Sequence violates learned vehicle physics (RPM/Speed Mismatch).")
else:
    print("\n[ ] Status: Data within normal physical bounds.")
print("="*40)