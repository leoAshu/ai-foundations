import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output


def plot_history(history, figsize=(7, 4)):
    # history keys are '<split>_<metric>', e.g. 'train_loss', 'val_acc';
    # each metric gets its own subplot with one line per split
    metrics = {}
    for key, values in history.items():
        split, metric = key.split('_', 1)
        metrics.setdefault(metric, []).append((split, values))

    clear_output(wait=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    for ax, (metric, lines) in zip(axes[0], metrics.items(), strict=True):
        for split, values in lines:
            ax.plot(range(1, len(values) + 1), values, label=split)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.show()

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
