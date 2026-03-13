import serial, jax, jax.numpy as jnp, numpy as np, pickle, time, re
from flax import linen as nn

# [Include same MiniGPT_IDS class as training script]

def run_ids():
    # Load the parameters
    with open('ids_model_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    model = MiniGPT_IDS(embed_dim=32, num_heads=4)
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
    
    # Threshold tuned from live testing
    ALERT_THRESHOLD = 0.20 
    
    # ... (rest of your logic for windowing and serial reading)