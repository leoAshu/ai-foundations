import pytest
import torch

from aif import DataModule


class TensorData(DataModule):
    def __init__(self, num_train=10, num_val=6, batch_size=4):
        super().__init__(batch_size=batch_size)

        self.num_train = num_train
        self.n = num_train + num_val

        self.X = torch.arange(self.n, dtype=torch.float32).reshape(-1, 1)
        self.y = self.X * 2

    def get_dataloader(self, train):
        indices = slice(0, self.num_train) if train else slice(self.num_train, self.n)

        return self._get_tensorloader((self.X, self.y), train, indices)


def test_get_dataloader_not_implemented():
    with pytest.raises(NotImplementedError):
        DataModule().train_dataloader()


def test_defaults():
    data = DataModule()

    assert data.root is None
    assert data.batch_size == 32


def test_train_val_split():
    data = TensorData()

    train_X = torch.cat([X for X, _ in data.train_dataloader()])
    val_X = torch.cat([X for X, _ in data.val_dataloader()])

    assert sorted(train_X.flatten().tolist()) == list(range(10))
    assert val_X.flatten().tolist() == list(range(10, 16))


def test_batches_keep_tensors_paired():
    for X, y in TensorData().train_dataloader():
        assert len(X) <= 4
        assert torch.equal(y, X * 2)
