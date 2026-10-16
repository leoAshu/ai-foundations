from pathlib import Path

import numpy as np
import pytest
import torch

from aif import BaseTimeMachine, TimeMachine, Vocab

DATA = Path(__file__).parents[1] / 'd2l' / 'data'


@pytest.fixture
def book(tmp_path):
    (tmp_path / 'timemachine.txt').write_text('The Time-Machine,\n by H. G. Wells 1895')

    return BaseTimeMachine(root=str(tmp_path))


def test_time_machine_default_root():
    assert BaseTimeMachine().root == '../data'


def test_time_machine_preprocess_keeps_lowercase_letters_and_spaces(book):
    assert book._preprocess(book._load()) == 'the time machine by h g wells '


def test_time_machine_tokenizes_characters(book):
    assert book._tokenize('ab c') == ['a', 'b', ' ', 'c']


def test_time_machine_build_maps_every_character(book):
    raw_text = book._load()
    corpus, vocab = book.build(raw_text)

    text = book._preprocess(raw_text)
    assert len(corpus) == len(text)
    assert vocab.to_tokens(corpus) == list(text)


def test_time_machine_build_reuses_given_vocab(book):
    vocab = Vocab(list('the'))
    corpus, returned = book.build(book._load(), vocab=vocab)

    assert returned is vocab
    # letters missing from the vocab map to <unk>
    assert vocab.unk in corpus


@pytest.mark.skipif(not (DATA / 'timemachine.txt').exists(), reason='no dataset')
def test_time_machine_full_book():
    data = BaseTimeMachine(root=str(DATA))
    corpus, vocab = data.build(data._load())

    # 26 letters, space and <unk>
    assert len(vocab) == 28
    assert len(corpus) > 100_000


@pytest.fixture
def long_book(tmp_path):
    # 'abcdefghij' repeated: 30 characters, all distinct within a period of 10
    (tmp_path / 'timemachine.txt').write_text('abcdefghij' * 3)

    return str(tmp_path)


def test_time_machine_windows_shift_targets_by_one(long_book):
    data = TimeMachine(batch_size=4, num_steps=5, root=long_book)

    # one window per start position that leaves room for the shifted target
    assert data.X.shape == data.Y.shape == (30 - 5, 5)
    assert data.vocab.to_tokens(data.X[0]) == list('abcde')
    assert data.vocab.to_tokens(data.Y[0]) == list('bcdef')
    assert (data.X[1:] == data.Y[:-1]).all()


def test_time_machine_train_and_val_split(long_book):
    data = TimeMachine(
        batch_size=4, num_steps=5, root=long_book, num_train=10, num_val=7
    )

    train, val = data.train_dataloader(), data.val_dataloader()

    assert data.batch_size == 4
    assert len(train.dataset) == 10
    assert len(val.dataset) == 7
    assert (val.dataset.tensors[0] == data.X[10:17]).all()


def test_time_machine_val_capped_by_available_windows(long_book):
    data = TimeMachine(batch_size=4, num_steps=5, root=long_book, num_train=20)

    # only 25 windows exist, so the default num_val of 5000 yields 5
    assert len(data.val_dataloader().dataset) == 5


def test_vocab_indices_are_sorted_tokens_with_unk():
    vocab = Vocab(['b', 'a', 'b'])

    assert vocab.idx_to_token == ['<unk>', 'a', 'b']
    assert len(vocab) == 3


def test_vocab_token_freqs_most_common_first():
    vocab = Vocab(['b', 'a', 'b', 'c', 'b', 'a'])

    assert vocab.token_freqs == [('b', 3), ('a', 2), ('c', 1)]


def test_vocab_lookup_single_nested_and_unknown():
    vocab = Vocab(['a', 'b'])

    assert vocab['a'] == 1
    assert vocab[['a', ('b', 'z')]] == [1, [2, vocab.unk]]
    assert vocab['z'] == vocab.unk == 0


def test_vocab_to_tokens():
    vocab = Vocab(['a', 'b'])

    assert vocab.to_tokens([1, 2]) == ['a', 'b']
    assert vocab.to_tokens(2) == 'b'


@pytest.mark.parametrize(
    ('indices', 'expected'),
    [
        ([2], ['b']),
        ((2,), ['b']),
        ([], []),
        (torch.tensor([2]), ['b']),
        (torch.tensor([1, 2]), ['a', 'b']),
        (np.array([2]), ['b']),
        (torch.tensor(2), 'b'),
        (np.int64(2), 'b'),
    ],
)
def test_vocab_to_tokens_sequences_and_scalars(indices, expected):
    assert Vocab(['a', 'b']).to_tokens(indices) == expected


def test_vocab_flattens_2d_tokens():
    assert Vocab([['a', 'b'], ['c']]).idx_to_token == ['<unk>', 'a', 'b', 'c']


def test_vocab_min_freq_and_reserved_tokens():
    vocab = Vocab(['a', 'a', 'b'], min_freq=2, reserved_tokens=['<pad>'])

    assert vocab.idx_to_token == ['<pad>', '<unk>', 'a']
    assert vocab['b'] == vocab.unk
