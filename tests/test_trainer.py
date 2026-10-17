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


def test_log_every_keeps_every_nth_and_last_epoch(regression_data, capsys):
    trainer = Trainer(max_epochs=7, plot=False, log_every=3)
    trainer.fit(LinearRegression(num_inputs=2), regression_data)

    lines = capsys.readouterr().out.splitlines()

    assert [line.split(' | ')[0] for line in lines] == [
        'Epoch 3/7',
        'Epoch 6/7',
        'Epoch 7/7',
    ]


def test_title_prefixes_log_line_and_pads_epoch(regression_data, capsys):
    trainer = Trainer(max_epochs=10, plot=False, title='Fold 2/5', log_every=5)
    trainer.fit(LinearRegression(num_inputs=2), regression_data)

    first = capsys.readouterr().out.splitlines()[0]

    assert first.startswith('Fold 2/5 | Epoch  5/10 | train_loss: ')


@pytest.fixture
def plot_calls(monkeypatch):
    calls = []

    def fake_plot_history(history, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr('aif.trainer.plot_history', fake_plot_history)

    return calls


def test_plot_called_each_epoch(regression_data, plot_calls):
    Trainer(max_epochs=3).fit(LinearRegression(num_inputs=2), regression_data)

    assert plot_calls == [{'xscale': 'linear', 'yscale': 'linear', 'title': None}] * 3


def test_plot_options_passed_through(regression_data, plot_calls):
    Trainer(max_epochs=1, yscale='log', title='Fold 1/5').fit(
        LinearRegression(num_inputs=2), regression_data
    )

    assert plot_calls == [{'xscale': 'linear', 'yscale': 'log', 'title': 'Fold 1/5'}]


def test_each_fit_gets_its_own_output_area(regression_data):
    trainer = Trainer(max_epochs=1, plot=False)
    first = (trainer._plot_display, trainer._status_display)

    trainer.fit(LinearRegression(num_inputs=2), regression_data)

    assert trainer._plot_display is not first[0]
    assert trainer._status_display is not first[1]


def grad_model(*grads):
    # a module whose parameters already hold the given gradients
    model = nn.Module()
    for i, grad in enumerate(grads):
        param = nn.Parameter(torch.zeros(len(grad)))
        param.grad = torch.tensor(grad)
        model.register_parameter(f'p{i}', param)

    return model


def grads(model):
    return [p.grad.tolist() for p in model.parameters()]


def test_clip_gradients_rescales_norm_above_threshold():
    model = grad_model([3.0, 4.0])

    Trainer(max_epochs=1).clip_gradients(1.0, model)

    assert grads(model) == [pytest.approx([0.6, 0.8])]


def test_clip_gradients_leaves_norm_below_threshold():
    model = grad_model([3.0, 4.0])

    Trainer(max_epochs=1).clip_gradients(5.0, model)

    assert grads(model) == [[3.0, 4.0]]


def test_clip_gradients_uses_norm_over_all_parameters():
    # norms 3 and 4 are each below 4.5, but together the norm is 5
    model = grad_model([3.0], [4.0])

    Trainer(max_epochs=1).clip_gradients(4.5, model)

    assert grads(model) == [pytest.approx([2.7]), pytest.approx([3.6])]


def test_clip_gradients_ignores_frozen_parameters():
    model = grad_model([3.0, 4.0], [100.0])
    model.p1.requires_grad_(False)

    Trainer(max_epochs=1).clip_gradients(1.0, model)

    assert grads(model) == [pytest.approx([0.6, 0.8]), [100.0]]


class RecordingSGD(SGD):
    # records the gradient norm each step sees
    def __init__(self, params, lr):
        super().__init__(params, lr)

        self.norms = []

    def step(self):
        grads = [p.grad.reshape(-1) for p in self.params]
        self.norms.append(torch.cat(grads).norm().item())

        super().step()


class RecordingLinearRegression(LinearRegression):
    def configure_optimizers(self):
        self.optim = RecordingSGD(self.parameters(), self.lr)

        return self.optim


def test_fit_clips_gradients_before_step(regression_data):
    model = RecordingLinearRegression(num_inputs=2)

    Trainer(max_epochs=2, plot=False, gradient_clip_val=0.5).fit(model, regression_data)

    # unclipped, the first steps on y = 2x1 - 3.4x2 + 4.2 have norms around 10
    assert max(model.optim.norms) == pytest.approx(0.5)


def test_fit_without_clipping_by_default(regression_data, monkeypatch):
    monkeypatch.setattr(
        Trainer, 'clip_gradients', lambda *args: pytest.fail('clip_gradients called')
    )
    model = RecordingLinearRegression(num_inputs=2)

    Trainer(max_epochs=1, plot=False).fit(model, regression_data)

    assert max(model.optim.norms) > 0.5
