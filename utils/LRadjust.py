import torch
from torch.optim import lr_scheduler

class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * ((1 - self.last_epoch / self.max_iter) ** self.power)
            for base_lr in self.base_lrs
        ]