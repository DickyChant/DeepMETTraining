import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class CyclicLR(_LRScheduler):
    """This scheduler implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle.
        max_lr (float or list): Upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
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
    
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.006,
        >>>                      step_size=2000, mode='triangular')
        >>> for epoch in range(100):
        >>>     for batch in train_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """
    
    def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', last_epoch=-1):
        
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        
        self.optimizer = optimizer
        
        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)
        
        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        
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
        
        super(CyclicLR, self).__init__(optimizer, last_epoch)
    
    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lrs = [new_base_lr] * len(self.optimizer.param_groups)
        if new_max_lr is not None:
            self.max_lrs = [new_max_lr] * len(self.optimizer.param_groups)
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
    
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.scale_mode == 'cycle':
            return self.base_lrs[0] + (self.max_lrs[0] - self.base_lrs[0]) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lrs[0] + (self.max_lrs[0] - self.base_lrs[0]) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
    
    def step(self, epoch=None):
        """Updates the learning rate for each parameter group.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_epoch(epoch)
        else:
            self.step_epoch(epoch)
    
    def step_epoch(self, epoch):
        """Updates the learning rate for each parameter group at the end of epoch.
        """
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def step_batch(self):
        """Updates the learning rate for each parameter group at the end of batch.
        """
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def get_lr(self):
        """Calculates the learning rate at batch iteration.
        """
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            lr = base_lr + (max_lr - base_lr) * self.clr()
            lrs.append(lr)
        return lrs
