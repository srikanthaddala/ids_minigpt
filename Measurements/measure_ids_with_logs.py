import serial
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import pickle
import time
import os
import csv

# --- 1. ARCHITECTURAL DIMENSIONS ---
BAUD_RATE = 115200
MODEL_PATH = 'ids_model_params.pkl'
LOG_FILE = 'ids_research_logs.csv'
INPUT_FEATURES = 5  
LAYER0_OUT = 32      
LAYER1_OUT = 64      
LAYER2_OUT = 32      

# --- 2. THE MODEL ---
class MiniGPT_IDS(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(LAYER0_OUT, name="Dense_0")(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(LAYER1_OUT, name="Dense_1")(x)
        x = nn.relu(x)
        x = nn.Dense(LAYER2_OUT, name="Dense_2")(x)
        return x

with open(MODEL_PATH, 'rb') as f:
    params = pickle.load(f)
model = MiniGPT_IDS()

@jax.jit
def predict_jit(params, x):
    return model.apply({'params': params}, x)

# --- 3. DATA PARSER ---
def parse_stm32_line(line):
    try:
        line_str = line.decode('utf-8')
        if "Data:" in line_str:
            data_part = line_str.split("Data:")[1].strip()
            hex_values = data_part.split()
            floats = [float(int(h, 16)) for h in hex_values[:INPUT_FEATURES]]
            return jnp.array(floats).reshape(1, 1, INPUT_FEATURES)
    except:
        return None
    return None

# --- 4. BENCHMARK & LOGGING ENGINE ---
def run_benchmark_with_logging():
    port = '/dev/ttyAMA0'
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"📡 System Live. Logging to {LOG_FILE}...")
    except:
        print("❌ Serial Port Busy.")
        return

    # Warm-up
    dummy = jnp.zeros((1, 1, INPUT_FEATURES))
    _ = predict_jit(params, dummy).block_until_ready()

    # Prepare CSV Header
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'inference_ms', 'loss_score', 'is_anomaly'])

    inf_latencies = []
    count = 0
    THRESHOLD = 0.5 # Example threshold for "Physical Sanity" violation

    print("🔥 Benchmark Started. Processing 200 samples...")

    try:
        while count < 200:
            line = ser.readline()
            if not line or b'Data:' not in line:
                continue
                
            x_actual = parse_stm32_line(line)
            if x_actual is None: continue

            # --- MEASURED INFERENCE ---
            t_inf_s = time.perf_counter_ns()
            # Predict what the NEXT values should be (or reconstruct current)
            prediction = predict_jit(params, x_actual).block_until_ready()
            t_inf_e = time.perf_counter_ns()
            # -------------------------

            # Calculate "Loss" (The Anomaly Score)
            # We compare a subset of the prediction to the input to find the 'Physical Gap'
            loss = float(jnp.mean(jnp.abs(prediction[:, :, :INPUT_FEATURES] - x_actual)))
            inf_ms = (t_inf_e - t_inf_s) / 1e6
            
            # Log Data
            inf_latencies.append(inf_ms)
            is_anomaly = 1 if loss > THRESHOLD else 0
            
            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), inf_ms, loss, is_anomaly])

            count += 1
            if count % 50 == 0:
                print(f"📊 Captured {count}/200 samples...")

    except KeyboardInterrupt:
        pass

    ser.close()
    print(f"\n✅ Mean Inference: {np.mean(inf_latencies):.4f} ms")
    print(f"📂 Raw research logs saved to {LOG_FILE}")

if __name__ == "__main__":
    run_benchmark_with_logging()
