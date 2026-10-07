import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from aif import plot_history, show_images


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def lines_by_label(ax):
    return {
        line.get_label(): np.asarray(line.get_ydata()).tolist()
        for line in ax.get_lines()
    }


def test_plot_history_one_subplot_per_metric():
    history = {
        'train_loss': [0.9, 0.5],
        'val_loss': [1.0, 0.6],
        'val_acc': [0.7, 0.8],
    }
    loss_ax, acc_ax = plot_history(history).axes

    assert loss_ax.get_ylabel() == 'loss'
    assert lines_by_label(loss_ax) == {'train': [0.9, 0.5], 'val': [1.0, 0.6]}
    assert acc_ax.get_ylabel() == 'acc'
    assert lines_by_label(acc_ax) == {'val': [0.7, 0.8]}


def test_plot_history_epochs_start_at_one():
    (ax,) = plot_history({'train_loss': [0.9, 0.5, 0.3]}).axes

    assert np.asarray(ax.get_lines()[0].get_xdata()).tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    ('shape', 'expected'),
    [((28, 28), (28, 28)), ((1, 28, 28), (28, 28)), ((3, 28, 28), (28, 28, 3))],
)
def test_show_images_tensor_shapes(shape, expected):
    (ax,) = show_images([torch.rand(shape)], nrows=1, ncols=1)

    assert ax.get_images()[0].get_array().shape == expected


def test_show_images_titles_and_spare_axes():
    axes = show_images(torch.rand(3, 1, 8, 8), nrows=2, ncols=2, titles=['a', 'b', 'c'])

    assert [ax.get_title() for ax in axes] == ['a', 'b', 'c', '']
    assert [len(ax.get_images()) for ax in axes] == [1, 1, 1, 0]


def test_show_images_default_grid():
    axes = show_images(torch.rand(10, 1, 8, 8))

    assert len(axes) == 8
    assert axes[0].figure.get_size_inches().tolist() == [12.0, 1.5]
