PhysicsGuard: Edge-Deployed Transformer IDS for Automotive CAN
PhysicsGuard is a Cyber-Physical Intrusion Detection System (IDS) that leverages a Mini-GPT Transformer to validate the "Physical Sanity" of vehicle signals. By reasoning about decoded signal vectors (Speed, RPM, Throttle) rather than just CAN IDs, it detects stealthy, sub-threshold attacks that bypass traditional security layers.

🏛️ System Architecture
Code snippet
graph TB
    subgraph "ZONE 1: TRAINING (Workstation)"
        A[Raw CAN Logs] --> B[JAX / Flax Model]
        B --> C[Training Loop]
        C --> D{ids_model_params.pkl}
    end

    subgraph "ZONE 2: HARDWARE (The Car)"
        E[STM32 Nucleo: ECU] -- "CAN Bus" --> F[MCP2515 Shield]
        F --> G[Arduino Uno: Bridge/Hacker]
    end

    subgraph "ZONE 3: DEPLOYMENT (Edge Hardware)"
        G -- "Serial/UART" --> H[measure_ids_transformer.py]
        D -. "Transfer" .-> I[In-Memory Weights]
        H --> J[Sliding Window Buffer]
        J & I --> K[JAX JIT Inference]
        K --> L{Anomaly Scorer}
    end
🚀 Research Highlights
Temporal Attention: Uses a Self-Attention mechanism with a 10-frame sliding context window to learn vehicle dynamics.
Sub-Millisecond Inference: Optimized with JAX and XLA (Ahead-of-Time) compilation, enabling a 1.04ms P99 latency on a Raspberry Pi 4.
44x Sensitivity Improvement: Detects "stealthy" injections—malicious data that stays within nominal sensor ranges but violates physical laws (e.g., sudden RPM spikes inconsistent with vehicle speed).
Decoupled Logic: Reasons about decoded physical signals rather than raw CAN IDs or hex bytes.
📊 Experimental Results (Performance Logs)
The following metrics were captured using Hardware-in-the-Loop (HIL) testing with a Raspberry Pi 4 Model B (8GB) and an STM32 Nucleo ECU simulator.

Metric	Result
Model Architecture	Decoder-only Mini-GPT (32-dim Embedding)
Mean Inference Latency	0.4318 ms
Worst-Case Latency (P99)	1.0418 ms
Detection Speed	~960 Hz (Exceeds 500kbps CAN requirements)
Anomaly Sensitivity	44x increase in MSE during injection
Detailed logs can be found in the /Measurements directory.

📂 Repository Structure
Plaintext
.
├── Measurements/
│   ├── transformer_research_logs.csv  # Raw inference & loss data
│   └── ids_performance_graph.png      # Visualization of anomaly spikes
├── deployment_pi/
│   ├── ids_model_params.pkl           # Trained model weights
│   └── measure_ids_transformer.py     # High-speed JAX inference engine
├── training_mac/
│   ├── ids_minigpt_train.py           # Transformer training logic
│   └── vehicle_data.csv               # Sample telemetry dataset
└── PhysicsGuard_Preprint.pdf          # Full Research Paper
🔌 Hardware Setup
ECU Simulator: STM32 Nucleo broadcasting Speed, RPM, and Gear telemetry over a 500kbps CAN Bus.
Hacker/Bridge: Arduino Uno + MCP2515 Shield acting as a gateway and an adversarial injection point.
IDS Node: Raspberry Pi 4 (8GB) running the JAX/XLA-optimized Transformer core.
🏃 How to Run
1. Training (Mac/PC)
Bash
cd training_mac
pip install jax flax numpy matplotlib
python3 ids_minigpt_train.py
2. Edge Deployment (Raspberry Pi)
Ensure the serial connection is established at /dev/ttyAMA0.

Bash
cd deployment_pi
python3 measure_ids_transformer.py
🏛️ Citation & Research Context
This work is part of an independent research project into low-latency Transformer applications for automotive security. For a detailed discussion of the methodology, please refer to the included Preprint PDF.

Author: Srikanth Addala

License: MIT
