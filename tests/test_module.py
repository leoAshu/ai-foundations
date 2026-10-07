import pytest
import torch
from torch import nn

from aif import Classifier, Module


class LinearRegression(Module):
    def __init__(self, num_inputs, lr=0.1):
        super().__init__(lr)

        self.w = nn.Parameter(torch.zeros(num_inputs, 1))
        self.b = nn.Parameter(torch.zeros(1))
        self.forward_calls = 0

    def forward(self, X):
        self.forward_calls += 1

        return X @ self.w + self.b

    def loss(self, y_hat, y):
        return ((y_hat - y) ** 2).mean()


class LinearClassifier(Classifier):
    def __init__(self, num_inputs, num_outputs, lr=0.1):
        super().__init__(lr)

        self.W = nn.Parameter(torch.zeros(num_inputs, num_outputs))

    def forward(self, X):
        return X @ self.W

    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y)


def test_base_forward_not_implemented():
    with pytest.raises(NotImplementedError):
        Module(lr=0.1)(torch.zeros(1))


def test_base_loss_not_implemented():
    with pytest.raises(NotImplementedError):
        Module(lr=0.1).loss(torch.zeros(1), torch.zeros(1))


def test_parameters_registered():
    model = LinearRegression(num_inputs=3)

    assert dict(model.named_parameters()).keys() == {'w', 'b'}


def test_steps_call_forward():
    model = LinearRegression(num_inputs=2)
    batch = (torch.ones(4, 2), torch.ones(4, 1))

    model.training_step(batch)
    model.validation_step(batch)

    assert model.forward_calls == 2


def test_validation_step_returns_loss_dict():
    model = LinearRegression(num_inputs=2)
    metrics = model.validation_step((torch.ones(4, 2), torch.ones(4, 1)))

    assert metrics.keys() == {'loss'}
    assert torch.isclose(metrics['loss'], torch.tensor(1.0))


def test_default_optimizer():
    model = LinearRegression(num_inputs=2, lr=0.05)
    optim = model.configure_optimizers()

    assert isinstance(optim, torch.optim.SGD)
    assert optim.param_groups[0]['lr'] == 0.05
    assert len(optim.param_groups[0]['params']) == 2


def test_training_reduces_loss():
    torch.manual_seed(0)
    X = torch.randn(64, 2)
    y = X @ torch.tensor([[2.0], [-3.4]]) + 4.2

    model = LinearRegression(num_inputs=2)
    optim = model.configure_optimizers()

    first = model.training_step((X, y)).item()
    for _ in range(50):
        optim.zero_grad()
        model.training_step((X, y)).backward()
        optim.step()

    assert model.training_step((X, y)).item() < first / 100


def test_accuracy():
    model = LinearClassifier(num_inputs=2, num_outputs=3)
    y_hat = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    y = torch.tensor([0, 2, 2])

    assert torch.isclose(model.accuracy(y_hat, y), torch.tensor(2 / 3))
    assert model.accuracy(y_hat, y, averaged=False).tolist() == [1.0, 0.0, 1.0]


def test_classifier_validation_step_returns_loss_and_acc():
    model = LinearClassifier(num_inputs=2, num_outputs=3)
    metrics = model.validation_step(
        (torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
    )

    assert metrics.keys() == {'loss', 'acc'}
