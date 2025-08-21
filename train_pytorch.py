#!/usr/bin/env python
# coding: utf-8

# PyTorch version - modified from Jan and Markus's code

import os
import pathlib
import datetime
import h5py
import optparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Local imports
from cyclical_learning_rate_pytorch import CyclicLR
from weighted_sum_layer_pytorch import WeightedSumLayer
from utils_pytorch import preProcessing, read_input, plot_history_pytorch
from loss_pytorch import custom_loss

class PTMissModel(nn.Module):
    def __init__(self, n_features=8, n_features_cat=3, n_dense_layers=3, 
                 activation='tanh', with_bias=False, maxNPF=4500, emb_out_dim=8):
        super(PTMissModel, self).__init__()
        
        self.n_features = n_features
        self.n_features_cat = n_features_cat
        self.n_dense_layers = n_dense_layers
        self.with_bias = with_bias
        self.maxNPF = maxNPF
        self.emb_out_dim = emb_out_dim
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(1000, emb_out_dim) for _ in range(n_features_cat)
        ])
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        input_size = n_features + n_features_cat * emb_out_dim
        
        for i in range(n_dense_layers):
            output_size = 8 * (2 ** (n_dense_layers - i))
            self.dense_layers.append(nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Tanh() if activation == 'tanh' else nn.ReLU(),
                nn.BatchNorm1d(output_size, momentum=0.95)
            ))
            input_size = output_size
        
        # Final dense layer
        final_output_size = 3 if with_bias else 1
        self.final_dense = nn.Linear(input_size, final_output_size)
        
        # Weighted sum layer
        self.weighted_sum = WeightedSumLayer(ndim=2, with_bias=with_bias)
        
        # Output layer (if with bias)
        if with_bias:
            self.output_layer = nn.Linear(2, 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.final_dense:
                    nn.init.normal_(m.weight, std=0.02)
                else:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0., std=0.4/self.emb_out_dim)
    
    def forward(self, x_cont, x_cat_list):
        batch_size = x_cont.size(0)
        
        # Extract px, py (last two features)
        pxpy = x_cont[:, :, -2:]
        
        # Process embeddings
        embeddings = []
        for i, (embedding_layer, x_cat) in enumerate(zip(self.embeddings, x_cat_list)):
            # Reshape for embedding: (batch, maxNPF, 1) -> (batch * maxNPF)
            x_cat_flat = x_cat.view(-1)
            emb = embedding_layer(x_cat_flat)
            emb = emb.view(batch_size, self.maxNPF, self.emb_out_dim)
            embeddings.append(emb)
        
        # Concatenate continuous features with embeddings
        x = torch.cat([x_cont] + embeddings, dim=2)
        
        # Process through dense layers
        for dense_layer in self.dense_layers:
            # Reshape for batch norm: (batch, maxNPF, features) -> (batch * maxNPF, features)
            x_reshaped = x.view(-1, x.size(-1))
            x_reshaped = dense_layer(x_reshaped)
            x = x_reshaped.view(batch_size, self.maxNPF, -1)
        
        # Final dense layer
        x = self.final_dense(x)
        
        # Concatenate with pxpy
        x = torch.cat([x, pxpy], dim=2)
        
        # Weighted sum layer
        x = self.weighted_sum(x)
        
        # Output layer if with bias
        if self.with_bias:
            x = self.output_layer(x)
        
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
                  help='device to use (cpu/cuda)', default='cpu', type='string')
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
device = torch.device(opt.device if torch.cuda.is_available() and opt.device == 'cuda' else 'cpu')
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
model = model.to(device)

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

# Convert to PyTorch tensors
Xi_train_tensor = torch.FloatTensor(Xi_train).to(device)
Xi_test_tensor = torch.FloatTensor(Xi_test).to(device)
Xc1_train_tensor = torch.LongTensor(Xc1_train).to(device)
Xc1_test_tensor = torch.LongTensor(Xc1_test).to(device)
Xc2_train_tensor = torch.LongTensor(Xc2_train).to(device)
Xc2_test_tensor = torch.LongTensor(Xc2_test).to(device)
Xc3_train_tensor = torch.LongTensor(Xc3_train).to(device)
Xc3_test_tensor = torch.LongTensor(Xc3_test).to(device)
Yr_train_tensor = torch.FloatTensor(Yr_train).to(device)
Yr_test_tensor = torch.FloatTensor(Yr_test).to(device)

# Create data loaders
train_dataset = TensorDataset(Xi_train_tensor, Xc1_train_tensor, Xc2_train_tensor, 
                             Xc3_train_tensor, Yr_train_tensor)
test_dataset = TensorDataset(Xi_test_tensor, Xc1_test_tensor, Xc2_test_tensor, 
                            Xc3_test_tensor, Yr_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Setup optimizer and learning rate scheduler
lr_scale = 1.
base_lr = 0.0003 * lr_scale
max_lr = 0.001 * lr_scale
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, 
                     step_size=len(Yr_train)//batch_size//2, mode='triangular2')

# Loss function
criterion = custom_loss

# Model summary
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Setup paths
if opt.load:
    timestamp = opt.load
else:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
path = f'models/{timestamp}'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# Load model if specified
if opt.load:
    model.load_state_dict(torch.load(f'{path}/model.pth', map_location=device))
    print(f'Restored model {timestamp}')

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

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_idx, (xi, xc1, xc2, xc3, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(xi, [xc1, xc2, xc3])
        loss = criterion(y, output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xi, xc1, xc2, xc3, y in test_loader:
            output = model(xi, [xc1, xc2, xc3])
            loss = criterion(y, output)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), f'{path}/best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'{path}/checkpoint_epoch_{epoch+1}.pth')

# Save final model
torch.save(model.state_dict(), f'{path}/final_model.pth')

# Plot training history
plot_history_pytorch(train_losses, val_losses, path)

print(f"Training completed. Model saved to {path}")
