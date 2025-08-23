import torch
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm up learning rate up to a target multiplier over a specified number of epochs,
    then optionally delegate to another scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier (float): Target LR multiplier after warmup (peak_lr = base_lr * multiplier).
        warm_epoch (int): Number of epochs for warmup.
        after_scheduler (_LRScheduler, optional): Scheduler to use after warmup.
    """
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        if multiplier < 1.0:
            raise ValueError("Multiplier should be >= 1.0.")
        self.multiplier = multiplier
        self.warm_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch <= self.warm_epoch:
            if self.multiplier == 1.0:
                # From 0 to base_lr
                return [base_lr * (self.last_epoch / self.warm_epoch) for base_lr in self.base_lrs]
            # Linear warmup
            return [base_lr * (1 + (self.multiplier - 1) * self.last_epoch / self.warm_epoch)
                    for base_lr in self.base_lrs]
        # After warmup
        if self.after_scheduler:
            if not self.finished:
                # adjust base_lrs of after_scheduler
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
            return self.after_scheduler.get_lr()
        # Hold peak LR
        return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            # Delegate stepping, offset epoch
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warm_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            # Default behavior
            return super().step(epoch)

# Example usage:
if __name__ == '__main__':
    # Suppose optimizer and epochs defined
    optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
    # Warmup for 5 epochs to 10x, then cosine decay
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, warm_epoch=5, after_scheduler=cosine)
    for epoch in range(100):
        # training loop ...
        scheduler.step()
        print(f"Epoch {epoch}: lr={optimizer.param_groups[0]['lr']}")