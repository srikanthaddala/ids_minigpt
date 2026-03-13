import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import pickle

# 1. Define the Mini-GPT Transformer Architecture
class MiniGPT_IDS(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, window_size, features)
        # Linear Embedding
        x = nn.Dense(self.embed_dim)(x)
        
        # Self-Attention Layer (The "Brain" of the sequence)
        attn_out = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = x + attn_out # Residual connection
        x = nn.LayerNorm()(x)
        
        # Feed Forward Network
        ff_out = nn.Dense(self.embed_dim * 2)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.embed_dim)(ff_out)
        x = x + ff_out
        
        # Predict the 5 features of the NEXT message based on the sequence
        return nn.Dense(5)(x[:, -1, :])

# 2. Setup Data and State
print("Loading data...")
X = jnp.load('X_train.npy')
y = jnp.load('y_train.npy')

# Initialize Model
model = MiniGPT_IDS(embed_dim=32, num_heads=4)
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 5)))
params = variables['params']

# Optimizer (Adam)
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply, 
    params=params, 
    tx=tx
)

# 3. Training Step (Fixed for modern Flax)
@jax.jit
def train_step(state, batch_X, batch_y):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, batch_X)
        return jnp.mean(jnp.square(pred - batch_y)) # MSE Loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Corrected method: apply_gradients
    state = state.apply_gradients(grads=grads) 
    return state, loss

# 4. Training Loop
print("Starting Training on Mac...")
for epoch in range(150): # Increased slightly for better convergence
    state, loss = train_step(state, X, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 5. Save the trained weights for the IDS script
with open('ids_model_params.pkl', 'wb') as f:
    pickle.dump(state.params, f)

print("-" * 30)
print("TRAINING COMPLETE")
print("Saved: ids_model_params.pkl")
print("The model now understands the 'Normal Grammar' of your car.")