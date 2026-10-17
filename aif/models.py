import torch
from torch import nn

from aif.module import Classifier, Module


def init_on_first_forward(lazy_module, init_fn):
    # a lazy module creates its weights in its own pre-hook on the first forward;
    # this one is registered after it, so init_fn runs once the weights exist
    # and before they are used
    @torch.no_grad()
    def hook(module, args):
        init_fn(module)
        handle.remove()

    handle = lazy_module.register_forward_pre_hook(hook)


def normal_init(sigma, bias_init=0.0):
    def init_fn(linear):
        linear.weight.normal_(0, sigma)
        linear.bias.fill_(bias_init)

    return init_fn


# defined in d2l/02_linear_nn/01_linear_regression.ipynb
class LinearRegression(Module):
    def __init__(self, lr, sigma=0.01, bias_init=0.0):
        super().__init__(lr)

        self.net = nn.LazyLinear(1)
        init_on_first_forward(self.net, normal_init(sigma, bias_init))

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)

    def get_w_b(self):
        return self.net.weight.detach(), self.net.bias.detach()


# defined in d2l/02_linear_nn/02_classification.ipynb; uses the loss of Classifier
class SoftmaxRegression(Classifier):
    def __init__(self, num_outputs, lr, sigma=0.01):
        super().__init__(lr)

        linear = nn.LazyLinear(num_outputs)
        init_on_first_forward(linear, normal_init(sigma))

        self.net = nn.Sequential(nn.Flatten(), linear)

        self.num_outputs = num_outputs

    def forward(self, X):
        return self.net(X)
