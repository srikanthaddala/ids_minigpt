import pandas as pd
import numpy as np

def prepare_ids_data(file_path):
    # 1. Load data
    df = pd.read_csv(file_path)
    
    # 2. Convert Hex strings to Integers
    for col in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']:
        df[col] = df[col].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)
    
    # 3. Calculate Delta-T (Crucial for Jitter detection)
    df['delta_t'] = df['timestamp'].diff().fillna(0)
    
    # 4. Map IDs to small integers (0, 1)
    id_map = {'0x101': 0, '0x102': 1}
    df['id_encoded'] = df['id'].map(id_map)
    
    # 5. Normalize values between 0 and 1
    # We only care about d0-d2 for our specific physics sim
    features = df[['id_encoded', 'd0', 'd1', 'd2', 'delta_t']].values
    features = features.astype(np.float32)
    
    # Simple Min-Max scaling for the bytes
    features[:, 1:4] /= 255.0 
    
    # 6. Create Sequences (Windows of 10 messages)
    window_size = 10
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])
        y.append(features[i+window_size]) # The target is the NEXT message
        
    return np.array(X), np.array(y)

# Usage
X_train, y_train = prepare_ids_data('vehicle_data.csv')
print(f"Prepared {len(X_train)} sequences for the Mini-GPT.")
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)