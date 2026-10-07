import pytest
import torch
from torch import nn

from aif import SGD, Classifier, DataModule, Module, Trainer


class TensorData(DataModule):
    def __init__(self, X, y, num_train, batch_size=16, with_val=True):
        super().__init__(batch_size=batch_size)

        self.X, self.y = X, y
        self.num_train = num_train
        self.with_val = with_val

    def get_dataloader(self, train):
        if not train and not self.with_val:
            return None

        indices = slice(0, self.num_train) if train else slice(self.num_train, None)

        return self._get_tensorloader((self.X, self.y), train, indices)


class LinearRegression(Module):
    def __init__(self, num_inputs, lr=0.1):
        super().__init__(lr)

        self.w = nn.Parameter(torch.zeros(num_inputs, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return X @ self.w + self.b

    def loss(self, y_hat, y):
        return ((y_hat - y) ** 2).mean()

    def configure_optimizers(self):
        return SGD(self.parameters(), self.lr)


class LinearClassifier(Classifier):
    def __init__(self, num_inputs, num_outputs, lr=0.5):
        super().__init__(lr)

        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, X):
        return self.linear(X)

    def loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y)


class MeanTarget(Module):
    # loss is the batch mean of y, so the epoch loss should be the mean of all y
    def __init__(self):
        super().__init__(lr=0.0)

        self.w = nn.Parameter(torch.zeros(1))
        self.modes = []

    def forward(self, X):
        self.modes.append(self.training)

        return X * self.w

    def loss(self, y_hat, y):
        return y.mean() + 0 * y_hat.sum()


@pytest.fixture
def regression_data():
    torch.manual_seed(0)
    X = torch.randn(200, 2)
    y = X @ torch.tensor([[2.0], [-3.4]]) + 4.2

    return TensorData(X, y, num_train=150)


def fit(model, data, max_epochs):
    trainer = Trainer(max_epochs=max_epochs, plot=False)
    trainer.fit(model, data)

    return trainer.history


def test_regression_history(regression_data):
    history = fit(LinearRegression(num_inputs=2), regression_data, max_epochs=5)

    assert history.keys() == {'train_loss', 'val_loss'}
    assert all(len(values) == 5 for values in history.values())
    assert history['val_loss'][-1] < history['val_loss'][0] / 100


def test_classifier_history():
    torch.manual_seed(0)
    X = torch.randn(300, 2)
    y = (X[:, 0] > X[:, 1]).long()

    history = fit(
        LinearClassifier(num_inputs=2, num_outputs=2),
        TensorData(X, y, num_train=200),
        max_epochs=10,
    )

    assert history.keys() == {'train_loss', 'val_loss', 'val_acc'}
    assert history['val_acc'][-1] > 0.9


def test_metrics_averaged_per_sample():
    X = torch.zeros(20, 1)
    y = torch.arange(20, dtype=torch.float32)

    # train: 0..9 in batches of 4, 4, 2; val: 10..19 in batches of 4, 4, 2
    history = fit(
        MeanTarget(), TensorData(X, y, num_train=10, batch_size=4), max_epochs=1
    )

    assert history['train_loss'] == [pytest.approx(4.5)]
    assert history['val_loss'] == [pytest.approx(14.5)]


def test_train_and_eval_modes():
    model = MeanTarget()
    data = TensorData(torch.zeros(8, 1), torch.zeros(8), num_train=4, batch_size=4)

    fit(model, data, max_epochs=1)

    assert model.modes == [True, False]


def test_without_val_dataloader(regression_data):
    regression_data.with_val = False

    history = fit(LinearRegression(num_inputs=2), regression_data, max_epochs=2)

    assert history.keys() == {'train_loss'}


def test_fit_resets_history(regression_data):
    trainer = Trainer(max_epochs=2, plot=False)

    trainer.fit(LinearRegression(num_inputs=2), regression_data)
    trainer.fit(LinearRegression(num_inputs=2), regression_data)

    assert all(len(values) == 2 for values in trainer.history.values())


def test_log_line(regression_data, capsys):
    fit(LinearRegression(num_inputs=2), regression_data, max_epochs=1)

    line = capsys.readouterr().out.strip()

    assert line.startswith('Epoch 1/1 | train_loss: ')
    assert ' | val_loss: ' in line


def test_plot_called_each_epoch(regression_data, monkeypatch):
    calls = []
    monkeypatch.setattr('aif.trainer.plot_history', calls.append)

    Trainer(max_epochs=3).fit(LinearRegression(num_inputs=2), regression_data)

    assert len(calls) == 3
