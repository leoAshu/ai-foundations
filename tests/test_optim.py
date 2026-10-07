import pytest
import torch

from aif import SGD, BaseOptimizer


def quadratic_loss(w):
    return ((w - torch.tensor([1.0, -2.0])) ** 2).sum()


@pytest.mark.parametrize('method', ['step', 'zero_grad'])
def test_base_not_implemented(method):
    optim = BaseOptimizer([torch.zeros(1, requires_grad=True)], lr=0.1)

    with pytest.raises(NotImplementedError):
        getattr(optim, method)()


def test_step_outside_no_grad():
    w = torch.zeros(2, requires_grad=True)
    optim = SGD([w], lr=0.1)

    quadratic_loss(w).backward()
    optim.step()

    assert w.requires_grad
    assert torch.allclose(w, torch.tensor([0.2, -0.4]))


def test_matches_torch_sgd():
    w = torch.zeros(2, requires_grad=True)
    w_ref = torch.zeros(2, requires_grad=True)

    optim = SGD([w], lr=0.1)
    optim_ref = torch.optim.SGD([w_ref], lr=0.1)

    for _ in range(5):
        for param, opt in ((w, optim), (w_ref, optim_ref)):
            opt.zero_grad()
            quadratic_loss(param).backward()
            opt.step()

    assert torch.allclose(w, w_ref)


def test_accepts_generator_of_params():
    model = torch.nn.Linear(2, 1)
    optim = SGD(model.parameters(), lr=0.1)

    for _ in range(2):
        optim.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        before = model.weight.clone()
        optim.step()

        assert not torch.equal(model.weight, before)


def test_zero_grad():
    w = torch.zeros(2, requires_grad=True)
    optim = SGD([w], lr=0.1)

    quadratic_loss(w).backward()
    optim.zero_grad()

    assert w.grad is not None
    assert torch.equal(w.grad, torch.zeros(2))


def test_skips_params_without_grad():
    w = torch.zeros(2, requires_grad=True)
    unused = torch.ones(2, requires_grad=True)
    optim = SGD([w, unused], lr=0.1)

    quadratic_loss(w).backward()
    optim.step()

    assert torch.equal(unused, torch.ones(2))
