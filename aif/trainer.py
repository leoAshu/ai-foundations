import matplotlib.pyplot as plt
import torch

from aif.plotting import LiveDisplay, plot_history


class Trainer:
    # log_every=None keeps one status line updated in place; log_every=N
    # keeps a line every N epochs (and the last epoch) instead;
    # gradient_clip_val > 0 clips the gradient norm to it, 0 disables clipping
    def __init__(
        self,
        max_epochs,
        plot=True,
        xscale='linear',
        yscale='linear',
        title=None,
        log_every=None,
        gradient_clip_val=0.0,
    ):
        self.max_epochs = max_epochs
        self.plot = plot
        self.xscale = xscale
        self.yscale = yscale
        self.title = title
        self.log_every = log_every
        self.gradient_clip_val = gradient_clip_val

        self._reset()

    # model, optim and dataloaders are set by fit()
    def _reset(self):
        self.epoch = 0
        self.history = {}

        # fresh output areas per fit, so earlier fits' output stays visible
        self._plot_display = LiveDisplay()
        self._status_display = LiveDisplay()

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model):
        self.model = model

    def prepare_batch(self, batch):
        return batch

    def fit(self, model, data):
        self._reset()
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.fit_epoch()

    def fit_epoch(self):
        metrics = {'train_loss': self._train_epoch()}

        if self.val_dataloader is not None:
            val_metrics = self._validate_epoch()
            metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)

        self.log(metrics)

    def log(self, metrics):
        if self.plot:
            fig = plot_history(
                self.history,
                xscale=self.xscale,
                yscale=self.yscale,
                title=self.title,
            )
            self._plot_display.update(fig)
            plt.close(fig)

        epoch = self.epoch + 1
        width = len(str(self.max_epochs))
        line = ' | '.join(
            ([self.title] if self.title else [])
            + [f'Epoch {epoch:>{width}}/{self.max_epochs}']
            + [f'{key}: {value:.6f}' for key, value in metrics.items()]
        )

        if self.log_every is None:
            self._status_display.update(line)
        elif epoch % self.log_every == 0 or epoch == self.max_epochs:
            print(line)

    # losses and metrics are batch means, so weight each by its batch size
    # to get a per-sample mean over the epoch
    def _train_epoch(self):
        self.model.train()
        total, num_samples = 0.0, 0

        for batch in self.train_dataloader:
            batch = self.prepare_batch(batch)
            loss = self.model.training_step(batch)

            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()

            total += loss.item() * len(batch[-1])
            num_samples += len(batch[-1])

        return total / num_samples

    # from d2l.ai 9.5.3: treats all parameters as one vector and rescales its
    # gradient so the norm is at most grad_clip_val
    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(torch.stack([torch.sum(p.grad**2) for p in params]).sum())
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    @torch.no_grad()
    def _validate_epoch(self):
        self.model.eval()
        totals, num_samples = {}, 0

        for batch in self.val_dataloader:
            batch = self.prepare_batch(batch)

            for key, value in self.model.validation_step(batch).items():
                totals[key] = totals.get(key, 0.0) + value.item() * len(batch[-1])
            num_samples += len(batch[-1])

        return {key: total / num_samples for key, total in totals.items()}
