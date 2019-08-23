from torch.optim.lr_scheduler import _LRScheduler



class CyclicalLR(_LRScheduler):
    """
    An implementation of Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/pdf/1506.01186.pdf

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of half lr adjustment cycle, see the paper for more details.
        max_lr (float): The maxium value of lr.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, step_size, max_lr, last_epoch=-1):
        self.step_size = step_size
        self.max_lr = max_lr
        super(CyclicalLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size * 2))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        return [
            base_lr + (self.max_lr - base_lr) * max(0, (1 - x)) * (1 / 2.**
                                                                   (cycle - 1))
            for base_lr in self.base_lrs
        ]



class CosineLR(_LRScheduler):
    """
    An implementation of SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS: https://arxiv.org/pdf/1608.03983.pdf

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        min_lr (float): The minimum value of lr. Default: 1e-5.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, step_size, min_lr=1e-5, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cur_step = (self.last_epoch % self.step_size) / self.step_size
        return [
            self.min_lr +
            (base_lr - self.min_lr) / 2 * (math.cos(math.pi * cur_step) + 1)
            for base_lr in self.base_lrs
        ]
