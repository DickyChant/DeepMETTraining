import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def read_input(inputfile):
    import h5py
    import os
    list_input = open("%s"%inputfile)
    nfiles = 0
    for line in list_input:
        fname = line.rstrip()
        if fname.startswith('#'):
            continue
        if not os.path.getsize(fname):
            continue
        print("read file", fname)
        h5f = h5py.File( fname, 'r')
        if nfiles == 0:
           X = h5f['X'][:]
           Y = h5f['Y'][:]
    
        else:
           X = np.concatenate((X, h5f['X']), axis=0)
           Y = np.concatenate((Y, h5f['Y']), axis=0)
        h5f.close()
        nfiles += 1
    
    print("finish reading files")
    return X, Y

def preProcessing(X, EVT=None):
    """ pre-processing input """
    norm = 50.0

    dxy = X[:,:,5:6]
    dz  = X[:,:,6:7].clip(-100, 100)
    eta = X[:,:,3:4]
    mass = X[:,:,8:9]
    pt = X[:,:,0:1] / norm
    puppi = X[:,:,7:8]
    px = X[:,:,1:2] / norm
    py = X[:,:,2:3] / norm

    # remove outliers
    pt[ np.where(np.abs(pt>200)) ] = 0.
    px[ np.where(np.abs(px>200)) ] = 0.
    py[ np.where(np.abs(py>200)) ] = 0.

    if EVT is not None:
        # environment variables
        evt = EVT[:,0:4]
        evt_expanded = np.expand_dims(evt, axis=1)
        evt_expanded = np.repeat(evt_expanded, X.shape[1], axis=1)
        # px py has to be in the last two columns
        inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, evt_expanded, px, py), axis=2)
    else:
        inputs = np.concatenate((dxy, dz, eta, mass, pt, puppi, px, py), axis=2)

    inputs_cat0 = X[:,:,11:12] # encoded PF pdgId
    inputs_cat1 = X[:,:,12:13] # encoded PF charge
    inputs_cat2 = X[:,:,13:14] # encoded PF fromPV

    return inputs, inputs_cat0, inputs_cat1, inputs_cat2

def plot_history_jax(train_losses, val_losses, path):
    """Plot training history for JAX training"""
    mpl.use('Agg')
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Training History (JAX)')
    plt.tight_layout()
    plt.savefig(f'{path}/history.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{path}/history.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save loss history to CSV
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    df.to_csv(f'{path}/loss_history.csv', index=False)
    
    print(f"Training history saved to {path}/")
