import jax
import jax.numpy as jnp
from flax import linen as nn

class WeightedSumLayer(nn.Module):
    '''Either does weight times inputs
    or weight times inputs + bias
    Input to be provided as:
      - Weights
      - ndim biases (if applicable)
      - ndim items to sum
    Currently works for 3-dim input, summing over the 2nd axis'''
    
    ndim: int = 2
    with_bias: bool = False
    
    def __call__(self, inputs):
        # input shape: B x E x F (batch x events x features)
        batch_size, n_events, n_features = inputs.shape
        
        # Extract weights (first column)
        weights = inputs[:, :, 0:1] - 1.0  # B x E x 1
        
        if not self.with_bias:
            # Extract features to sum (all except weights)
            tosum = inputs[:, :, 1:]  # B x E x (F-1)
            # Apply weights: weights * tosum (broadcasting)
            weighted = weights * tosum  # B x E x (F-1)
        else:
            # Extract biases and features to sum
            biases = inputs[:, :, 1:self.ndim+1]  # B x E x ndim
            tosum = inputs[:, :, self.ndim+1:]  # B x E x (F-ndim-1)
            # Apply weights: weights * (biases + tosum)
            weighted = weights * (biases + tosum)  # B x E x (F-ndim-1)
        
        # Sum over the events dimension (axis=1)
        return jnp.sum(weighted, axis=1)  # B x (F-1) or B x (F-ndim-1)
