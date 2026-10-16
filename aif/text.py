import collections
import re
from pathlib import Path

import torch

from aif.data import DataModule


# defined in d2l/05_rnn/02_exploratory_language_analysis.ipynb as TimeMachine
class BaseTimeMachine(DataModule):
    root: str

    def __init__(self, root='../data'):
        super().__init__(root)

    def _load(self):
        file = Path(self.root) / 'timemachine.txt'

        with open(file) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_text, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None:
            vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]

        return corpus, vocab


# defined in d2l/05_rnn/03_language_sequential_data_prep.ipynb
class TimeMachine(BaseTimeMachine):
    def __init__(
        self, batch_size, num_steps, root='../data', num_train=10_000, num_val=5_000
    ):
        super().__init__(root)

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val

        corpus, self.vocab = self.build(self._load())

        array = torch.tensor(
            [corpus[i : i + num_steps + 1] for i in range(len(corpus) - num_steps)]
        )

        self.X, self.Y = array[:, :-1], array[:, 1:]

    def get_dataloader(self, train):
        idx = (
            slice(0, self.num_train)
            if train
            else slice(self.num_train, self.num_train + self.num_val)
        )

        return self._get_tensorloader([self.X, self.Y], train, idx)


# defined in d2l/05_rnn/02_exploratory_language_analysis.ipynb
class Vocab:
    # the list defaults are only read, never mutated, so sharing them is safe
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):  # noqa: B006
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # The list of unique tokens
        self.idx_to_token = list(
            sorted(
                set(
                    ['<unk>']
                    + reserved_tokens
                    + [token for token, freq in self.token_freqs if freq >= min_freq]
                )
            )
        )
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        # any sequence, even of length 1, gives a list; a 0-d tensor or array
        # has __len__ but no length, so it counts as a single index
        if hasattr(indices, '__len__') and getattr(indices, 'ndim', 1) > 0:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[int(indices)]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
