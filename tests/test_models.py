import pytest
import torch
from torch import nn

from aif import DataModule, LinearRegression, SoftmaxClassifier, Trainer


class SyntheticData(DataModule):
    def __init__(self, w, b, num_train=800, num_val=200, batch_size=32):
        super().__init__(batch_size=batch_size)

        self.num_train = num_train
        self.X = torch.randn(num_train + num_val, len(w))
        self.y = self.X @ w.reshape(-1, 1) + b

    def get_dataloader(self, train):
        indices = slice(0, self.num_train) if train else slice(self.num_train, None)

        return self._get_tensorloader((self.X, self.y), train, indices)


def test_linear_regression_infers_num_inputs():
    model = LinearRegression(lr=0.1)
    y_hat = model(torch.ones(4, 3))
    w, _ = model.get_w_b()

    assert y_hat.shape == (4, 1)
    assert w.shape == (1, 3)


def test_linear_regression_init_on_first_forward():
    torch.manual_seed(0)
    model = LinearRegression(lr=0.1, sigma=0.01)
    model(torch.ones(1, 1000))
    w, b = model.get_w_b()

    # LazyLinear's own init would give std ~1/sqrt(3 * 1000) ~ 0.018 and a nonzero bias
    assert w.std().item() == pytest.approx(0.01, rel=0.1)
    assert b.tolist() == [0.0]
    assert not w.requires_grad and not b.requires_grad


def test_linear_regression_init_runs_once():
    model = LinearRegression(lr=0.1)
    model(torch.ones(1, 2))
    with torch.no_grad():
        model.net.bias.fill_(5.0)
    model(torch.ones(1, 2))

    assert model.get_w_b()[1].tolist() == [5.0]


def test_linear_regression_loss_is_mse():
    model = LinearRegression(lr=0.1)
    y_hat = torch.tensor([[1.0], [3.0]])
    y = torch.tensor([[0.0], [1.0]])

    assert model.loss(y_hat, y).item() == pytest.approx(2.5)


def test_linear_regression_recovers_parameters():
    torch.manual_seed(0)
    w, b = torch.tensor([2.0, -3.4]), 4.2
    model = LinearRegression(lr=0.03)

    Trainer(max_epochs=10, plot=False).fit(model, SyntheticData(w, b))

    w_hat, b_hat = model.get_w_b()
    assert w_hat.reshape(-1).tolist() == pytest.approx(w.tolist(), abs=1e-3)
    assert b_hat.item() == pytest.approx(b, abs=1e-3)


def test_linear_regression_bias_init():
    model = LinearRegression(lr=0.1, bias_init=12.0)
    model(torch.ones(1, 2))

    assert model.get_w_b()[1].tolist() == [12.0]


def test_softmax_classifier_flattens_and_infers_num_inputs():
    model = SoftmaxClassifier(num_outputs=10, lr=0.1)
    y_hat = model(torch.ones(4, 1, 28, 28))

    linear = model.net[1]
    assert isinstance(linear, nn.Linear)  # LazyLinear becomes Linear once built
    assert y_hat.shape == (4, 10)
    assert linear.weight.shape == (10, 784)


def test_softmax_classifier_init_on_first_forward():
    torch.manual_seed(0)
    model = SoftmaxClassifier(num_outputs=10, lr=0.1, sigma=0.01)
    model(torch.ones(1, 1000))
    linear = model.net[1]

    assert isinstance(linear, nn.Linear)
    assert linear.weight.std().item() == pytest.approx(0.01, rel=0.1)
    assert linear.bias.detach().abs().max().item() == 0.0


def test_softmax_classifier_loss_reshapes_and_reduces():
    model = SoftmaxClassifier(num_outputs=3, lr=0.1)
    Y_hat = torch.zeros(2, 2, 3)
    Y = torch.zeros(2, 2, dtype=torch.long)

    # uniform logits over 3 classes: cross-entropy is log(3) per prediction
    assert model.loss(Y_hat, Y).item() == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )
    assert model.loss(Y_hat, Y, averaged=False).shape == (4,)


def test_softmax_classifier_learns():
    torch.manual_seed(0)
    X = torch.randn(300, 2)
    y = (X[:, 0] > X[:, 1]).long()

    class Data(DataModule):
        def get_dataloader(self, train):
            indices = slice(0, 200) if train else slice(200, None)

            return self._get_tensorloader((X, y), train, indices)

    trainer = Trainer(max_epochs=10, plot=False)
    trainer.fit(SoftmaxClassifier(num_outputs=2, lr=0.5), Data(batch_size=16))

    assert trainer.history['val_acc'][-1] > 0.9
