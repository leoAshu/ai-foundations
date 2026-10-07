import torch


class BaseOptimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(BaseOptimizer):
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
