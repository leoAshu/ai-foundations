import matplotlib.pyplot as plt
import torch
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Pretty, display
from matplotlib_inline.backend_inline import set_matplotlib_formats


class LiveDisplay:
    # one output area that is replaced in place on every update, so output
    # displayed before it (e.g. earlier folds) is kept
    def __init__(self):
        self._handle = None

    def update(self, obj):
        if not InteractiveShell.initialized():
            # plain python: nothing to update in place, so just print text
            if isinstance(obj, str):
                print(obj)
            return

        if isinstance(obj, str):
            obj = Pretty(obj)

        if self._handle is None:
            self._handle = display(obj, display_id=True)
        else:
            self._handle.update(obj)


def use_svg_display():
    # IPython.display.set_matplotlib_formats (used by d2l) was removed in IPython 9;
    # the replacement creates an IPython shell if none exists, which would make
    # LiveDisplay think it runs in a notebook, so only call it inside one
    if InteractiveShell.initialized():
        set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale='linear',
    yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'),
    figsize=(3.5, 2.5),
    axes=None,
):
    # port of d2l.plot; X and Y are one curve or a list of curves, a single X is
    # shared by every Y, and with Y=None the values of X are plotted against
    # their index; legend labels the curves in order
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # True if X (tensor or list) has 1 axis
    def has_one_axis(X):
        return (
            hasattr(X, 'ndim')
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], '__len__')
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):  # noqa: B905 - curves beyond fmts are dropped, as in d2l
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def plot_history(history, figsize=(7, 4), xscale='linear', yscale='linear', title=None):
    # history keys are '<split>_<metric>', e.g. 'train_loss', 'val_acc';
    # each metric gets its own subplot with one line per split
    metrics = {}
    for key, values in history.items():
        split, metric = key.split('_', 1)
        metrics.setdefault(metric, []).append((split, values))

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    for ax, (metric, lines) in zip(axes[0], metrics.items(), strict=True):
        for split, values in lines:
            ax.plot(range(1, len(values) + 1), values, label=split)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig


def show_images(imgs, nrows=1, ncols=8, titles=None, scale=1.5):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * scale, nrows * scale), squeeze=False
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis('off')
        if i >= len(imgs):
            continue

        img = imgs[i]
        if torch.is_tensor(img):
            img = img.detach().cpu()
            # (C, H, W) -> (H, W) for grayscale, (H, W, C) for colour
            if img.ndim == 3:
                img = img.squeeze(0) if img.shape[0] == 1 else img.permute(1, 2, 0)
            img = img.numpy()

        ax.imshow(img)
        if titles:
            ax.set_title(titles[i])

    return axes
