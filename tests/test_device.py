import pytest
import torch

from aif import cpu, gpu, num_gpus, try_gpu


def test_cpu_and_gpu_devices():
    assert cpu() == torch.device('cpu')
    assert gpu() == torch.device('cuda:0')
    assert gpu(1) == torch.device('cuda:1')


def test_num_gpus_matches_torch():
    assert num_gpus() == torch.cuda.device_count()


@pytest.mark.parametrize(
    ('available', 'i', 'expected'),
    [(0, 0, 'cpu'), (1, 0, 'cuda:0'), (1, 1, 'cpu'), (2, 1, 'cuda:1')],
)
def test_try_gpu_falls_back_to_cpu(monkeypatch, available, i, expected):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: available)

    assert try_gpu(i) == torch.device(expected)
