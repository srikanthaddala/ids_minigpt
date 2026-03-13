import jax
import jax.numpy as jnp
from flax import linen as nn
import pickle

# Define the architecture exactly as it will be on the Pi
class MiniGPT_IDS(nn.Module):
    embed_dim: int
    num_heads: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        attn_out = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = x + attn_out
        x = nn.LayerNorm()(x)
        ff_out = nn.Dense(self.embed_dim * 2)(x)
        ff_out = nn.relu(ff_out)
        ff_out = nn.Dense(self.embed_dim)(ff_out)
        x = x + ff_out
        return nn.Dense(5)(x[:, -1, :])

# Add a note: This script saves 'ids_model_params.pkl'