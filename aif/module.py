import torch
from torch import nn

from aif.optim import BaseOptimizer


class Module(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

    def forward(self, X):
        raise NotImplementedError

    def loss(self, y_hat, y):
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer | BaseOptimizer:
        return torch.optim.SGD(self.parameters(), self.lr)

    def training_step(self, batch):
        y_hat = self(*batch[:-1])

        return self.loss(y_hat, batch[-1])

    def validation_step(self, batch):
        y_hat = self(*batch[:-1])

        return {'loss': self.loss(y_hat, batch[-1])}


class Classifier(Module):
    def accuracy(self, y_hat, y, averaged=True):
        preds = y_hat.argmax(1)
        compare = (preds == y).float()

        return compare.mean() if averaged else compare

    # from d2l.ai 4.5.2, so every classifier gets it
    def loss(self, y_hat, y, averaged=True):
        # reshaped to support multi-dimensional predictions & targets
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape(-1)

        return nn.functional.cross_entropy(
            y_hat, y, reduction='mean' if averaged else 'none'
        )

    def validation_step(self, batch):
        y = batch[-1]
        y_hat = self(*batch[:-1])

        return {'loss': self.loss(y_hat, y), 'acc': self.accuracy(y_hat, y)}
