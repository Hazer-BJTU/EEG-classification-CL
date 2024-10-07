import torch
import torch.nn as nn
from models import *


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


class GRUblock(nn.Module):
    def __init__(self, input_size, hiddens, layers, dropout, **kwargs):
        super(GRUblock, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.layers = layers
        self.dropout = dropout
        self.network = nn.GRU(input_size, hiddens, num_layers=layers, batch_first=True,
                              dropout=dropout, bidirectional=True)

    def get_initial_states(self, batch_size, device):
        return torch.zeros((2 * self.layers, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, seq_length = X.shape[0], X.shape[1]
        H0 = self.get_initial_states(batch_size, X.device)
        (output, Hn) = self.network(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class Upsample(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, **kwargs):
        super(Upsample, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.hiddens = hiddens
        self.output_channels = output_channels
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hiddens[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(), nn.BatchNorm1d(hiddens[0]),
            nn.Conv1d(hiddens[0], hiddens[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(hiddens[1]),
            nn.Conv1d(hiddens[1], hiddens[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm1d(hiddens[2]),
            nn.Conv1d(hiddens[2], output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, X):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(-1, F, T)
        X = self.network(X)
        X = X.view(batch_size, seq_length, self.output_channels, T)
        return X


class Encoder(nn.Module):
    def __init__(self, dropout=0.25, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.filter_banks = FilterBanks(258, 128, 64, 25)
        self.dropout1 = nn.Dropout(dropout)
        self.short_term_gru = ShortTermGRU(64, 128, 2, dropout)
        self.attention = Attention(256)
        self.long_term_gru = LongTermGRU(256, 128, 2, dropout)
        self.label2vec = Label2Vec(256)
        self.mu = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        self.sigma = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128)
        )

    def forward(self, X, y):
        batch_size, seq_length = X.shape[0], X.shape[1]
        X = self.filter_banks(X)
        X = self.dropout1(X)
        X = self.short_term_gru(X)
        X = self.attention(X)
        X = self.long_term_gru(X)
        y = self.label2vec(y)
        y = y.view(-1, y.shape[2])
        Z = torch.cat((X, y), dim=1)
        output1, output2 = self.mu(Z), self.sigma(Z)
        return output1.view(batch_size, seq_length, -1), output2.view(batch_size, seq_length, -1)


class Decoder(nn.Module):
    def __init__(self, dropout=0.25, channels=2, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.rnn = GRUblock(256, 300, 2, dropout)
        self.label2vec = Label2Vec(128)
        self.upsample = Upsample(24, (64, 128, 256), 129 * channels)

    def forward(self, X, y):
        batch_size, seq_length = X.shape[0], X.shape[1]
        y = self.label2vec(y)
        X = torch.cat((X, y), dim=2)
        X = self.rnn(X)
        X = X.view(batch_size, seq_length, X.shape[2] // 25, 25)
        X = self.upsample(X)
        return X


class CVAE(nn.Module):
    def __init__(self, dropout=0.25, channels=2, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = Encoder(dropout)
        self.decoder = Decoder(dropout, channels)

    def forward(self, X, y, z):
        mu, sigma = self.encoder(X, y)
        Z = mu + z * sigma.exp()
        X_fake = self.decoder(Z, y)
        return X_fake, mu, sigma


if __name__ == '__main__':
    X = torch.randn((16, 10, 258, 25), dtype=torch.float32, requires_grad=False, device='cpu')
    y = torch.randint(0, 5, (16, 10), dtype=torch.int64, requires_grad=False, device='cpu')
    z = torch.randn((16, 10, 128), dtype=torch.float32, requires_grad=False, device='cpu')
    net = CVAE()
    X_fake, mu, sigma = net(X, y, z)
    print(X_fake.shape, mu.shape, sigma.shape)
    torch.save(net.state_dict(), 'cvae_network.pth')
    