

### 📂 Repository Structure

```text
/can-ids-minigpt
├── README.md
├── requirements.txt
├── hardware/
│   ├── stm32_ecu_sim.cpp
│   └── uno_hacker_node.cpp
├── model/
│   ├── ids_minigpt_train.py
│   └── ids_model_params.pkl  (You'll upload this)
└── pi_deployment/
    └── live_ids.py

```

---

### 1. 📄 README.md

T
 **Mermaid diagram** (GitHub renders this automatically!) and the project overview.

```markdown
# Real-Time CAN Bus IDS using Mini-GPT and JAX

A Cyber-Physical Intrusion Detection System (IDS) that uses a Transformer (Mini-GPT) to detect anomalies on a vehicle's CAN Bus.

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "ZONE 1: TRAINING (Mac Studio)"
        A[Raw CAN Logs] --> B[JAX / Flax Model]
        B --> C[Training Loop]
        C --> D{ids_model_params.pkl}
    end

    subgraph "ZONE 2: HARDWARE (The Car)"
        E[STM32 Nucleo: ECU] -- "CAN Bus" --> F[MCP2515 Shield]
        F --> G[Arduino Uno: Bridge/Hacker]
    end

    subgraph "ZONE 3: DEPLOYMENT (Raspberry Pi)"
        G -- "Serial" --> H[live_ids.py]
        D -. "Transfer" .-> I[In-Memory Weights]
        H --> J[Sliding Window Buffer]
        J & I --> K[JAX JIT Inference]
        K --> L{Anomaly Scorer}
    end

```

## 🚀 Overview

* **Training:** Performed on Mac using **JAX/Flax** for high-speed sequence learning.
* **Hardware:** STM32 simulates vehicle physics; Arduino Uno acts as a gateway and potential injection point.
* **Detection:** Raspberry Pi 4 runs real-time inference using **JIT-compiled JAX code**.
* **The Concept:** The model learns the "Laws of Physics" of the car. When a hacker injects a fake speed (e.g., 200km/h), the prediction error spikes, triggering an alert.

## 🛠️ Installation

```bash
pip install jax[cpu] flax optax pyserial numpy

```

```

---

### 2. 📝 requirements.txt
```text
jax[cpu]
flax
optax
pyserial
numpy

```

---

### 3. 🐍 live_ids.py (The Pi Guardian)

```python
import serial, jax, jax.numpy as jnp, numpy as np, pickle, time, re
from flax import linen as nn

class MiniGPT_IDS(nn.Module):
    embed_dim: int; num_heads: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        x = x + nn.SelfAttention(num_heads=self.num_heads)(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(5)(x[:, -1, :])

# Initialization
with open('ids_model_params.pkl', 'rb') as f:
    params = pickle.load(f)
model = MiniGPT_IDS(embed_dim=32, num_heads=4)
ser = serial.Serial('/dev/serial0', 115200, timeout=1)
pattern = re.compile(r"ID=(0x[0-9A-F]+) \| Data: (([0-9A-F]{2}\s?){8})")
window = []; last_time = time.time()

print("IDS ACTIVE...")
try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        match = pattern.search(line)
        if match:
            can_id = match.group(1)
            data_bytes = match.group(2).strip().split()
            dt = time.time() - last_time; last_time = time.time()
            new_entry = np.array([1 if can_id=='0x102' else 0, int(data_bytes[0],16)/255, int(data_bytes[1],16)/255, int(data_bytes[2],16)/255, dt], dtype=np.float32)
            
            if len(window) == 10:
                input_seq = jnp.array(window).reshape(1, 10, 5)
                pred = model.apply({'params': params}, input_seq)
                error = jnp.mean(jnp.square(pred - new_entry))
                status = "!!! ATTACK !!!" if error > 0.15 else "NORMAL"
                print(f"[{status}] ID: {can_id} | Error: {error:.4f}")
                window.pop(0)
            window.append(new_entry)
except KeyboardInterrupt: pass

```

---

### 4. 🤖 uno_hacker_node.cpp (The Attack)

```cpp
#include <SPI.h>
#include <mcp2515.h>
struct can_frame canMsg;
MCP2515 mcp2515(10);

void setup() {
  Serial.begin(115200);
  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
  mcp2515.setNormalMode();
}

void loop() {
  // Inject Attack every 15 seconds
  if (millis() % 15000 < 500) { 
    canMsg.can_id  = 0x102; // Speed ID
    canMsg.can_dlc = 8;
    canMsg.data[0] = 0xC8; // 200 km/h (The Anomaly)
    mcp2515.sendMessage(&canMsg);
  } 
  // Standard Receive Logic...
}

```

Hardware Setup

To replicate this environment, you need three nodes on the CAN network:

ECU Simulator (STM32 Nucleo): Acts as the primary ECU, broadcasting legitimate vehicle telemetry (Speed, RPM, Gear).

Hacker/Bridge Node (Arduino Uno + MCP2515): Acts as the gateway. It listens to the bus and passes data to the Pi via Serial. It also contains the "Attack" code to inject anomalies.

IDS Node (Raspberry Pi 4): Connected to the Arduino via USB/Serial. Runs the JAX-based detection model.

Connection Diagram:
STM32 (CAN_H/L) <---> Arduino/MCP2515 (CAN_H/L) <---> Raspberry Pi (USB/Serial)

🏃 How to Run

1. Training (on your Workstation)

If you want to re-train the model or use your own data:

Bash
cd training_mac
pip install -r ../requirements_mac.txt
python3 train_ids.py
This generates ids_model_params.pkl.

2. Deployment (on Raspberry Pi)

Ensure the Arduino is plugged into the Pi and the .pkl file is in the deployment_pi folder.

Bash
cd deployment_pi
pip install -r ../requirements_pi.txt
python3 live_ids.py
3. Simulating an Attack

Trigger the attack code on the Arduino. You should see the terminal output on the Pi switch from NORMAL to !!! ATTACK !!! with a corresponding jump in the MSE error score.

💡 Pro-Tip for the hardware

If a pen-tester looks at this, they will ask about the Serial Baud Rate. In your live_ids.py, you set it to 115200. It’s grounded to mention that this is the current bottleneck—if you were to move to a dedicated CAN Hat for the Pi, you could skip the Arduino bridge and increase the sampling frequency.

