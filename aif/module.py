import torch
from torch import nn


class Module(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

    def forward(self, X):
        raise NotImplementedError

    def loss(self, y_hat, y):
        raise NotImplementedError

    def configure_optimizers(self):
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

    def validation_step(self, batch):
        y = batch[-1]
        y_hat = self(*batch[:-1])

        return {'loss': self.loss(y_hat, y), 'acc': self.accuracy(y_hat, y)}
