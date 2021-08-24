import torch
from torch import nn


class FirstParser(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_features: int,
        num_tokens: int,
        hidden_dim: int,
        num_transitions: int,
        dropout_prob=0.5,
    ):
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        """
        super(FirstParser, self).__init__()
        self._activation = torch.tanh
        self.embed_size = embedding_dim
        self.n_features = n_features

        self.embeddings = nn.Embedding(num_tokens, embedding_dim)
        self.embed_to_hidden = nn.Linear(n_features * embedding_dim, hidden_dim)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight, gain=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_to_logits = nn.Linear(hidden_dim, num_transitions)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight, gain=1)

    def embedding_lookup(self, t):
        embedded = self.embeddings(t)
        x = embedded.view(t.size(0), self.n_features * self.embed_size)
        return x

    def forward(self, t):
        emb = self.embedding_lookup(t)
        hidden = self.embed_to_hidden(emb)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.hidden_to_logits(hidden)
        return logits
