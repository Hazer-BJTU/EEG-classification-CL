import torch
import torch.nn as nn


class Label2Vec(nn.Module):
    def __init__(self, features, **kwargs):
        super(Label2Vec, self).__init__(**kwargs)
        self.features = features
        self.block = nn.Linear(5, features)

    def forward(self, X):
        X_one_hot = torch.nn.functional.one_hot(X, 5)
        batch_size, seq_length = X_one_hot.shape[0], X_one_hot.shape[1]
        X_one_hot = X_one_hot.to(torch.float32)
        if not X_one_hot.is_contiguous():
            X_one_hot = X_one_hot.contiguous()
        X_one_hot = X_one_hot.view(batch_size * seq_length, 5)
        output = self.block(X_one_hot)
        output = output.view(batch_size, seq_length, self.features)
        return output


class GenerativeGRU(nn.Module):
    def __init__(self, input_size, hiddens, layers, dropout, mean=0, std=1, **kwargs):
        super(GenerativeGRU, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.layers = layers
        self.dropout = dropout
        self.mean, self.std = mean, std
        self.network = nn.GRU(input_size, hiddens, num_layers=layers, batch_first=True,
                              dropout=dropout, bidirectional=True)

    def get_initial_states(self, batch_size, device):
        return torch.normal(self.mean, self.std, (2 * self.layers, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        H0 = self.get_initial_states(batch_size, X.device)
        (output, Hn) = self.network(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class LinearNetwork(nn.Module):
    def __init__(self, input_features, length, hiddens, output_features, dropout, **kwargs):
        super(LinearNetwork, self).__init__(**kwargs)
        self.input_features = input_features
        self.length = length
        self.hiddens = hiddens
        self.output_features = output_features
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.Conv1d(input_features, hiddens, kernel_size=1, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hiddens, hiddens, kernel_size=1, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hiddens, output_features, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, X):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        assert features % self.length == 0
        X = X.view(batch_size * seq_length, features // self.length, self.length)
        output = self.block(X)
        output = output.view(batch_size, seq_length, self.output_features, self.length)
        return output


class GenerativeNetwork(nn.Module):
    def __init__(self, dropout, channels=2, **kwargs):
        super(GenerativeNetwork, self).__init__(**kwargs)
        self.dropout = dropout
        self.channels = channels
        self.label2vec = Label2Vec(64)
        self.rnn = GenerativeGRU(64, 200, 2, dropout)
        self.linear = LinearNetwork(16, 25, 128, 129 * channels, dropout)

    def forward(self, X):
        X = self.label2vec(X)
        X = self.rnn(X)
        X = self.linear(X)
        return X


if __name__ == '__main__':
    net = GenerativeNetwork(0.25)
    X = torch.randint(0, 5, (16, 10))
    print(X.shape)
    print(net(X).shape)
    torch.save(net.state_dict(), 'generative_network.pth')
