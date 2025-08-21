#!/usr/bin/env python
# coding: utf-8

# JAX version - modified from Jan and Markus's code

import os
import pathlib
import datetime
import h5py
import optparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from flax import linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from cyclical_learning_rate_jax import CyclicLR
from weighted_sum_layer_jax import WeightedSumLayer
from utils_jax import preProcessing, read_input, plot_history_jax
from loss_jax import custom_loss

class PTMissModel(nn.Module):
    n_features: int = 8
    n_features_cat: int = 3
    n_dense_layers: int = 3
    activation: str = 'tanh'
    with_bias: bool = False
    maxNPF: int = 4500
    emb_out_dim: int = 8
    
    @nn.compact
    def __call__(self, x_cont, x_cat_list, training=True):
        batch_size = x_cont.shape[0]
        
        # Extract px, py (last two features)
        pxpy = x_cont[:, :, -2:]
        
        # Process embeddings
        embeddings = []
        for i in range(self.n_features_cat):
            embedding_layer = nn.Embed(num_embeddings=1000, features=self.emb_out_dim)
            x_cat = x_cat_list[i]
            # Reshape for embedding: (batch, maxNPF, 1) -> (batch * maxNPF)
            x_cat_flat = x_cat.reshape(-1)
            emb = embedding_layer(x_cat_flat)
            emb = emb.reshape(batch_size, self.maxNPF, self.emb_out_dim)
            embeddings.append(emb)
        
        # Concatenate continuous features with embeddings
        x = jnp.concatenate([x_cont] + embeddings, axis=2)
        
        # Process through dense layers
        for i in range(self.n_dense_layers):
            output_size = 8 * (2 ** (self.n_dense_layers - i))
            x = nn.Dense(output_size)(x)
            
            if self.activation == 'tanh':
                x = jnp.tanh(x)
            else:
                x = jax.nn.relu(x)
            
            x = nn.BatchNorm(use_running_statistics=not training, momentum=0.95)(x)
        
        # Final dense layer
        final_output_size = 3 if self.with_bias else 1
        x = nn.Dense(final_output_size)(x)
        
        # Concatenate with pxpy
        x = jnp.concatenate([x, pxpy], axis=2)
        
        # Weighted sum layer
        x = WeightedSumLayer(ndim=2, with_bias=self.with_bias)(x)
        
        # Output layer if with bias
        if self.with_bias:
            x = nn.Dense(2)(x)
        
        return x

def create_model(n_features=8, n_features_cat=3, n_dense_layers=3, 
                activation='tanh', with_bias=False, maxNPF=4500, emb_out_dim=8):
    model = PTMissModel(
        n_features=n_features, 
        n_features_cat=n_features_cat, 
        n_dense_layers=n_dense_layers,
        activation=activation, 
        with_bias=with_bias,
        maxNPF=maxNPF,
        emb_out_dim=emb_out_dim
    )
    return model

# Configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input',
                  help='input file', default='tree_100k.h5', type='string')
parser.add_option('-l', '--load', dest='load',
                  help='load model from timestamp', default='', type='string')
parser.add_option('--nfiles', dest='nfiles', 
                  help='number of h5df files', default=100, type='int')
parser.add_option('--withbias', dest='withbias',
                  help='include bias term in the DNN', default=False, action="store_true")
parser.add_option('--device', dest='device',
                  help='device to use (cpu/gpu/tpu)', default='cpu', type='string')
(opt, args) = parser.parse_args()

# General setup
maxNPF = 4500
n_features_pf = 8
n_features_pf_cat = 3
normFac = 50.
epochs = 100
batch_size = 64
preprocessed = True
emb_out_dim = 8

# Device setup
if opt.device == 'gpu' and jax.devices('gpu'):
    device = jax.devices('gpu')[0]
elif opt.device == 'tpu' and jax.devices('tpu'):
    device = jax.devices('tpu')[0]
else:
    device = jax.devices('cpu')[0]

print(f"Using device: {device}")

##
## Read input and do preprocessing
##
Xorg, Y = read_input(opt.input)
Y = Y / -normFac

Xi, Xc1, Xc2, Xc3 = preProcessing(Xorg)
print(f"Xi shape: {Xi.shape}")
print(f"Xc1 shape: {Xc1.shape}")
print(f"Xc2 shape: {Xc2.shape}")
print(f"Xc3 shape: {Xc3.shape}")
print(f"maxNPF: {maxNPF}")
print(f"n_features: {n_features_pf}")

Xc = [Xc1, Xc2, Xc3]
emb_input_dim = {
    i: int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
}
print('Embedding input dimensions', emb_input_dim)

# Create model
model = create_model(
    n_features=n_features_pf, 
    n_features_cat=n_features_pf_cat, 
    with_bias=opt.withbias,
    maxNPF=maxNPF,
    emb_out_dim=emb_out_dim
)

# Prepare training/val data
Yr = Y
Xr = [Xi] + Xc
indices = np.array([i for i in range(len(Yr))])
indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=7)

# Split data into train/test sets
Xi_train, Xi_test = Xi[indices_train], Xi[indices_test]
Xc1_train, Xc1_test = Xc1[indices_train], Xc1[indices_test]
Xc2_train, Xc2_test = Xc2[indices_train], Xc2[indices_test]
Xc3_train, Xc3_test = Xc3[indices_train], Xc3[indices_test]
Yr_train = Yr[indices_train]
Yr_test = Yr[indices_test]

# Convert to JAX arrays
Xi_train_jax = jnp.array(Xi_train)
Xi_test_jax = jnp.array(Xi_test)
Xc1_train_jax = jnp.array(Xc1_train, dtype=jnp.int32)
Xc1_test_jax = jnp.array(Xc1_test, dtype=jnp.int32)
Xc2_train_jax = jnp.array(Xc2_train, dtype=jnp.int32)
Xc2_test_jax = jnp.array(Xc2_test, dtype=jnp.int32)
Xc3_train_jax = jnp.array(Xc3_train, dtype=jnp.int32)
Xc3_test_jax = jnp.array(Xc3_test, dtype=jnp.int32)
Yr_train_jax = jnp.array(Yr_train)
Yr_test_jax = jnp.array(Yr_test)

# Initialize model parameters
key = jax.random.PRNGKey(0)
dummy_input_cont = jnp.ones((1, maxNPF, n_features_pf))
dummy_input_cat = [jnp.ones((1, maxNPF, 1), dtype=jnp.int32) for _ in range(n_features_pf_cat)]
params = model.init(key, dummy_input_cont, dummy_input_cat)['params']

# Setup optimizer and learning rate scheduler
lr_scale = 1.
base_lr = 0.0003 * lr_scale
max_lr = 0.001 * lr_scale

# Create optimizer
optimizer = optax.adam(learning_rate=base_lr)
opt_state = optimizer.init(params)

# Create learning rate scheduler
scheduler = CyclicLR(base_lr=base_lr, max_lr=max_lr, 
                     step_size=len(Yr_train)//batch_size//2, mode='triangular2')

# Loss function
def loss_fn(params, batch):
    xi, xc1, xc2, xc3, y = batch
    x_cat_list = [xc1, xc2, xc3]
    pred = model.apply({'params': params}, xi, x_cat_list)
    return custom_loss(y, pred)

# Gradient function
grad_fn = jit(grad(loss_fn))

# Training step
@jit
def train_step(params, opt_state, batch):
    grads = grad_fn(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Evaluation step
@jit
def eval_step(params, batch):
    xi, xc1, xc2, xc3, y = batch
    x_cat_list = [xc1, xc2, xc3]
    pred = model.apply({'params': params}, xi, x_cat_list)
    loss = custom_loss(y, pred)
    return loss

# Model summary
print(model)
total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"Total parameters: {total_params:,}")

# Setup paths
if opt.load:
    timestamp = opt.load
else:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
path = f'models/{timestamp}'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# Save model summary
with open(f'{path}/summary.txt', 'w') as txtfile:
    txtfile.write(str(model) + '\n')
    txtfile.write(f"Total parameters: {total_params:,}\n")

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Create batches
def create_batches(xi, xc1, xc2, xc3, y, batch_size):
    n_samples = len(xi)
    indices = jnp.arange(n_samples)
    indices = jax.random.permutation(indices)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_xi = xi[batch_indices]
        batch_xc1 = xc1[batch_indices]
        batch_xc2 = xc2[batch_indices]
        batch_xc3 = xc3[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_xi, batch_xc1, batch_xc2, batch_xc3, batch_y))
    
    return batches

for epoch in range(epochs):
    # Training phase
    train_batches = create_batches(Xi_train_jax, Xc1_train_jax, Xc2_train_jax, 
                                  Xc3_train_jax, Yr_train_jax, batch_size)
    
    train_loss = 0.0
    for batch in train_batches:
        params, opt_state = train_step(params, opt_state, batch)
        train_loss += eval_step(params, batch)
    
    train_loss /= len(train_batches)
    train_losses.append(float(train_loss))
    
    # Validation phase
    val_batches = create_batches(Xi_test_jax, Xc1_test_jax, Xc2_test_jax, 
                                Xc3_test_jax, Yr_test_jax, batch_size)
    
    val_loss = 0.0
    for batch in val_batches:
        val_loss += eval_step(params, batch)
    
    val_loss /= len(val_batches)
    val_losses.append(float(val_loss))
    
    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        with open(f'{path}/best_model.npz', 'wb') as f:
            jnp.savez(f, **params)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        with open(f'{path}/checkpoint_epoch_{epoch+1}.npz', 'wb') as f:
            jnp.savez(f, **params)

# Save final model
with open(f'{path}/final_model.npz', 'wb') as f:
    jnp.savez(f, **params)

# Plot training history
plot_history_jax(train_losses, val_losses, path)

print(f"Training completed. Model saved to {path}")
