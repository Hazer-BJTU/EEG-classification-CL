import copy
import torch
import torch.nn as nn


class FilterBanks(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, features, **kwargs):
        super(FilterBanks, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.hiddens = hiddens
        self.output_channels = output_channels
        self.features = features
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1, padding=0),
            nn.ReLU(), nn.BatchNorm1d(hiddens, track_running_stats=False),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(), nn.BatchNorm1d(output_channels, track_running_stats=False)
        )

    def forward(self, X):
        if not X.is_contiguous():
            X = X.contiguous()
        batch_size, window_size = X.shape[0], X.shape[1]
        X = X.view(-1, self.input_channels, self.features)
        X = self.network(X)
        X = X.transpose(1, 2).view(batch_size, window_size, self.features, self.output_channels)
        return X


class ShortTermGRU(nn.Module):
    def __init__(self, input_size, hiddens, layers, dropout, **kwargs):
        super(ShortTermGRU, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.dropout = dropout
        self.layers = layers
        self.network = nn.GRU(input_size, hiddens, num_layers=layers, batch_first=True,
                              dropout=dropout, bidirectional=True)

    def get_initial_states(self, batch_size, device):
        return torch.zeros((2 * self.layers, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, window_size, length = X.shape[0], X.shape[1], X.shape[2]
        X = X.view(-1, length, self.input_size)
        H0 = self.get_initial_states(batch_size * window_size, X.device)
        (output, Hn) = self.network(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        output = output.view(batch_size, window_size, length, 2 * self.hiddens)
        return output


class Attention(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.input_size = input_size
        self.Watt = nn.Parameter(torch.randn((input_size, 1)))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X):
        batch_size, window_size, length = X.shape[0], X.shape[1], X.shape[2]
        X = X.view(-1, length, self.input_size)
        A = torch.matmul(X, self.Watt)
        A = A.transpose(1, 2)
        A = self.softmax(A)
        X = torch.bmm(A, X)
        X = torch.squeeze(X, dim=1)
        if not X.is_contiguous():
            X = X.contiguous
        X = X.view(batch_size, window_size, -1)
        return X


class LongTermGRU(nn.Module):
    def __init__(self, input_size, hiddens, layers, dropout, **kwargs):
        super(LongTermGRU, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.layers = layers
        self.dropout = dropout
        self.network = nn.GRU(input_size, hiddens, num_layers=layers, batch_first=True,
                              dropout=dropout, bidirectional=True)

    def get_initial_states(self, batch_size, device):
        return torch.zeros((2 * self.layers, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, length = X.shape[0], X.shape[1]
        H0 = self.get_initial_states(batch_size, X.device)
        (output, Hn) = self.network(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        output = output.view(-1, self.hiddens * 2)
        return output


def init_weight(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif hasattr(module, 'weight'):
        nn.init.normal_(module.weight)


class SeqSleepNet(nn.Module):
    def __init__(self, dropout=0.25, **kwargs):
        super(SeqSleepNet, self).__init__(**kwargs)
        self.filter_banks = FilterBanks(258, 128, 64, 25)
        self.dropout1 = nn.Dropout(dropout)
        self.short_term_gru = ShortTermGRU(64, 256, 2, dropout)
        self.attention = Attention(512)
        self.long_term_gru = LongTermGRU(512, 256, 2, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 5)
        )

    def forward(self, X):
        X = self.filter_banks(X)
        X = self.dropout1(X)
        X = self.short_term_gru(X)
        X = self.attention(X)
        X = self.long_term_gru(X)
        X = self.classifier(X)
        return X


if __name__ == '__main__':
    net = SeqSleepNet()
    X = torch.zeros(16, 10, 258, 25)
    print(X.shape)
    y = net(X)
    print(y.shape)
