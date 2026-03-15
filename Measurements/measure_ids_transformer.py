import serial
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import pickle
import time
import os
import csv

# --- 1. CONFIGURATION ---
BAUD_RATE = 115200
LOG_FILE = 'transformer_research_logs.csv'
INPUT_FEATURES = 5  
CONTEXT_WINDOW = 10 
EMBED_DIM = 32      

# --- 2. THE MINI-GPT ARCHITECTURE (Option B: Corrected with Projection Head) ---
class MiniGPT_IDS(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, seq_len, features) -> (1, 10, 5)
        
        # 1. Input Projection (5 -> 32)
        x = nn.Dense(EMBED_DIM, name="Embedding_Layer")(x)
        
        # 2. Self-Attention Layer
        attn_out = nn.SelfAttention(num_heads=4, name="Attention_Block")(x)
        x = x + attn_out 
        x = nn.LayerNorm()(x)
        
        # 3. Feed Forward Network
        ff_out = nn.Dense(EMBED_DIM * 2, name="FFN_Expansion")(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(EMBED_DIM, name="FFN_Projection")(ff_out)
        x = x + ff_out 
        
        # 4. FINAL DECODING HEAD (32 -> 5)
        # This fixes the "incompatible shapes for broadcasting" error
        return nn.Dense(INPUT_FEATURES, name="Output_Projection")(x)

# --- 3. INITIALIZATION ---
model = MiniGPT_IDS()
key = jax.random.PRNGKey(42)
# Initialize with the context window shape
variables = model.init(key, jnp.zeros((1, CONTEXT_WINDOW, INPUT_FEATURES)))
params = variables['params']

@jax.jit
def predict_jit(params, x):
    return model.apply({'params': params}, x)

def parse_stm32_line(line):
    try:
        line_str = line.decode('utf-8')
        if "Data:" in line_str:
            data_part = line_str.split("Data:")[1].strip()
            hex_values = data_part.split()
            floats = [float(int(h, 16)) for h in hex_values[:INPUT_FEATURES]]
            return jnp.array(floats)
    except:
        return None
    return None

# --- 4. BENCHMARK & LOGGING ---
def run_transformer_benchmark():
    port = '/dev/ttyAMA0'
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"📡 Transformer Ingestion Active. Logging to {LOG_FILE}...")
    except:
        print("❌ Port Busy. Check connections.")
        return

    # Warm-up 
    print("🧠 Compiling JAX XLA Attention Graph (10-frame context)...")
    dummy_context = jnp.zeros((1, CONTEXT_WINDOW, INPUT_FEATURES))
    _ = predict_jit(params, dummy_context).block_until_ready()

    # Prepare CSV
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'inference_ms', 'anomaly_score'])

    inf_latencies = []
    context_buffer = [] 

    print("🔥 Starting Transformer Benchmark. Collecting 100 Sequences...")

    count = 0
    try:
        while count < 100:
            line = ser.readline()
            if not line or b'Data:' not in line:
                continue
                
            new_frame = parse_stm32_line(line)
            if new_frame is None: continue

            context_buffer.append(new_frame)
            if len(context_buffer) > CONTEXT_WINDOW:
                context_buffer.pop(0)
            
            if len(context_buffer) == CONTEXT_WINDOW:
                x_input = jnp.array(context_buffer).reshape(1, CONTEXT_WINDOW, INPUT_FEATURES)

                # --- MEASURED TRANSFORMER INFERENCE ---
                t_inf_s = time.perf_counter_ns()
                prediction = predict_jit(params, x_input).block_until_ready()
                t_inf_e = time.perf_counter_ns()
                # --------------------------------------

                inf_ms = (t_inf_e - t_inf_s) / 1e6
                inf_latencies.append(inf_ms)
                
                # Now shapes match: (1, 5) - (1, 5)
                # Prediction is for the last frame in the window
                current_pred = prediction[0, -1, :] 
                current_actual = x_input[0, -1, :]
                loss = float(jnp.mean(jnp.abs(current_pred - current_actual)))

                with open(LOG_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), inf_ms, loss])

                count += 1
                if count % 25 == 0:
                    print(f"📊 Captured {count}/100 Transformer sequences...")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")

    ser.close()
    
    if inf_latencies:
        print("\n" + "="*50)
        print("🏛️ REAL TRANSFORMER RESULTS (Attention Core)")
        print("="*50)
        print(f"Inference Latency (Mean): {np.mean(inf_latencies):.4f} ms")
        print(f"Worst Case (P99):        {np.percentile(inf_latencies, 99):.4f} ms")
        print(f"Architecture:            Transformer (Mini-GPT)")
        print(f"Log File:                {LOG_FILE}")
        print("="*50)

if __name__ == "__main__":
    run_transformer_benchmark()
