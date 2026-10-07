import torch


class DataModule:
    def __init__(self, root=None, batch_size=32):
        self.root = root
        self.batch_size = batch_size

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(True)

    def val_dataloader(self):
        return self.get_dataloader(False)

    def _get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(t[indices] for t in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)

        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
