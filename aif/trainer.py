import torch

from aif.plot import plot_history


class Trainer:
    def __init__(self, max_epochs, plot=True):
        self.max_epochs = max_epochs
        self.plot = plot

        self._reset()

    # model, optim and dataloaders are set by fit()
    def _reset(self):
        self.epoch = 0
        self.history = {}

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
            plot_history(self.history)

        print(
            ' | '.join(
                [f'Epoch {self.epoch + 1}/{self.max_epochs}']
                + [f'{key}: {value:.6f}' for key, value in metrics.items()]
            )
        )

    # losses and metrics are batch means, so weight each by its batch size
    # to get a per-sample mean over the epoch
    def _train_epoch(self):
        self.model.train()
        total, num_samples = 0.0, 0

        for batch in self.train_dataloader:
            batch = self.prepare_batch(batch)
            loss = self.model.training_step(batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total += loss.item() * len(batch[-1])
            num_samples += len(batch[-1])

        return total / num_samples

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
