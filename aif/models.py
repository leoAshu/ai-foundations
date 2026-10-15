import torch
from torch import nn

from aif.module import Module


# defined in d2l/02_linear_nn/01_linear_regression.ipynb
class LinearRegression(Module):
    def __init__(self, lr, sigma=0.01):
        super().__init__(lr)

        self.sigma = sigma
        self.net = nn.LazyLinear(1)

        # LazyLinear creates its weights in its own pre-hook on the first forward;
        # this one is registered after it, so it runs once the weights exist
        self._init_handle = self.net.register_forward_pre_hook(self._init_params)

    @torch.no_grad()
    def _init_params(self, module, args):
        self.net.weight.normal_(0, self.sigma)
        self.net.bias.fill_(0)

        self._init_handle.remove()

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)

    def get_w_b(self):
        return self.net.weight.detach(), self.net.bias.detach()
