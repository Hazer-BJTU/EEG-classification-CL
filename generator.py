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
        return torch.zeros((2 * self.layers, batch_size, self.hiddens), device=device)

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
        self.rnn = GenerativeGRU(128, 225, 2, dropout)
        self.linear = LinearNetwork(18, 25, 128, 129 * channels, dropout)
        self.tanh = nn.Tanh()

    def forward(self, X, noise):
        X = self.label2vec(X)
        X = torch.cat((X, noise), dim=2)
        X = self.rnn(X)
        X = self.linear(X)
        X = self.tanh(X)
        return X


class AntiLinear(nn.Module):
    def __init__(self, input_features, hiddens, output_features, dropout, **kwargs):
        super(AntiLinear, self).__init__(**kwargs)
        self.input_features = input_features
        self.output_features = output_features
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.Conv1d(input_features, hiddens, kernel_size=5, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hiddens, hiddens, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hiddens, output_features, kernel_size=3, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

    def forward(self, X):
        batch_size, length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        if not X.is_contiguous():
            X = X.contiguous
        X = X.view(-1, F, T)
        output = self.block(X)
        output = output.view(batch_size, length, self.output_features, -1)
        return output


class DiscrimitiveGRU(nn.Module):
    def __init__(self, input_size, hiddens, layers, dropout, **kwargs):
        super(DiscrimitiveGRU, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.layers = layers
        self.dropout = dropout
        self.network = nn.GRU(input_size, hiddens, num_layers=layers, batch_first=True,
                              dropout=dropout, bidirectional=True)

    def get_initial_states(self, batch_size, device):
        return torch.zeros((2 * self.layers, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        H0 = self.get_initial_states(batch_size, X.device)
        (output, Hn) = self.network(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class DiscrimitiveNetwork(nn.Module):
    def __init__(self, dropout, channels, **kwargs):
        super(DiscrimitiveNetwork, self).__init__(**kwargs)
        self.antilinear = AntiLinear(129 * channels, 64, 16, dropout)
        self.label2vec = Label2Vec(64)
        self.rnn = DiscrimitiveGRU(128, 172, 2, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(344, 172),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(172, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, X, y):
        Y = self.label2vec(y)
        X = self.antilinear(X).view(X.shape[0], X.shape[1], -1)
        X = torch.cat((X, Y), dim=2)
        X = self.rnn(X)
        X = X.view(-1, X.shape[2])
        X = self.classifier(X)
        return X


if __name__ == '__main__':
    net = GenerativeNetwork(0.1, 2)
    X = torch.randint(0, 5, (16, 10), dtype=torch.int64, device='cpu')
    noise = torch.randn(16, 10, 64)
    print(net(X, noise).shape)
    torch.save(net.state_dict(), 'generative_network.pth')
    torch.save(DiscrimitiveNetwork(0.1, 2).state_dict(), 'discrimitive_network.pth')
