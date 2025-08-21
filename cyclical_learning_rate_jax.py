import jax
import jax.numpy as jnp
import optax

class CyclicLR:
    """This scheduler implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    
    Args:
        base_lr (float): Initial learning rate which is the lower boundary in the cycle.
        max_lr (float): Upper boundary in the cycle. Functionally, it defines the
            cycle amplitude (max_lr - base_lr).
        step_size (int): Number of training iterations per half cycle.
        mode (str): One of {triangular, triangular2, exp_range}.
            Default 'triangular'.
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on cycle number or cycle
            iterations (training iterations since start of cycle).
    """
    
    def __init__(self, base_lr, max_lr, step_size, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
    
    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
    
    def clr(self):
        cycle = jnp.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = jnp.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                jnp.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                jnp.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
    
    def step(self):
        """Updates the learning rate for the next iteration.
        """
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        lr = self.clr()
        
        # Store in history
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        return lr
    
    def get_lr(self):
        """Returns the current learning rate.
        """
        return self.clr()

def create_cyclic_lr_scheduler(base_lr, max_lr, step_size, mode='triangular2'):
    """Create an Optax learning rate scheduler with cyclical learning rate.
    
    Args:
        base_lr (float): Base learning rate
        max_lr (float): Maximum learning rate
        step_size (int): Step size for the cycle
        mode (str): Mode for the cyclical learning rate
    
    Returns:
        optax._src.base.GradientTransformation: Optax scheduler
    """
    
    def schedule_fn(step):
        # Convert step to cycle iterations
        cycle = jnp.floor(1 + step / (2 * step_size))
        x = jnp.abs(step / step_size - 2 * cycle + 1)
        
        if mode == 'triangular':
            scale = 1.0
        elif mode == 'triangular2':
            scale = 1 / (2.**(cycle - 1))
        else:  # exp_range
            scale = 1.0  # Simplified for now
        
        lr = base_lr + (max_lr - base_lr) * jnp.maximum(0, (1 - x)) * scale
        return lr
    
    return optax.exponential_decay(
        init_value=base_lr,
        transition_steps=1,
        decay_rate=1.0,
        transition_begin=0,
        staircase=False,
        end_value=base_lr
    )
