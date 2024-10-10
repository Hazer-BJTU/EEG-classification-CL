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


class GRUlayer(nn.Module):
    def __init__(self, input_size, hiddens, layers, dropout=0, **kwargs):
        super(GRUlayer, self).__init__(**kwargs)
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


class CNNlayer(nn.Module):
    def __init__(self, channels_lst, kernels_lst, **kwargs):
        super(CNNlayer, self).__init__(**kwargs)
        self.channels_lst = channels_lst
        self.kernels_lst = kernels_lst
        assert len(channels_lst) > 1
        self.block = nn.Sequential()
        for idx in range(0, len(channels_lst) - 1):
            self.block.add_module(f'module_#{idx}_conv',
                                  nn.Conv1d(channels_lst[idx], channels_lst[idx+1],
                                            kernel_size=kernels_lst[idx], stride=1, padding='same'))
            if idx != len(channels_lst) - 2:
                self.block.add_module(f'module_#{idx}_relu', nn.ReLU())

    def forward(self, X):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, F, T)
        X = self.block(X)
        X = X.view(batch_size, seq_length, -1, T)
        return X


class Encoder(nn.Module):
    def __init__(self, channels_num, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.label2vec = Label2Vec(64)
        self.cnn = CNNlayer([channels_num * 129, 128, 128, 64, 32], [5, 5, 5, 5])
        self.linear = nn.Sequential(nn.Linear(864, 512), nn.ReLU())
        self.rnn = GRUlayer(512, 256, 2)
        self.mu = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        self.sigma = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        self.invariant = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))

    def forward(self, X, y):
        y = self.label2vec(y)
        X = self.cnn(X).view(X.shape[0], X.shape[1], -1)
        X = torch.cat((X, y), dim=2)
        X = self.linear(X)
        X = self.rnn(X)
        output1, output2, output3 = self.mu(X), self.sigma(X), self.invariant(X)
        return output1, output2, output3


class Decoder(nn.Module):
    def __init__(self, channels_num, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.label2vec = Label2Vec(64)
        self.rnn = GRUlayer(448, 256, 2)
        self.linear = nn.Sequential(nn.Linear(512, 800), nn.ReLU())
        self.cnn = CNNlayer([32, 64, 128, 256, channels_num * 129], [5, 5, 5, 5])

    def forward(self, X, y):
        y = self.label2vec(y)
        X = torch.cat((X, y), dim=2)
        X = self.rnn(X)
        X = self.linear(X)
        X = X.view(X.shape[0], X.shape[1], 32, 25)
        X = self.cnn(X)
        return X


class CVAE(nn.Module):
    def __init__(self, channels_num, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = Encoder(channels_num)
        self.decoder = Decoder(channels_num)

    def forward(self, X, y, z):
        mu, sigma, invariant = self.encoder(X, y)
        Z = mu + sigma.exp() * z
        Z = torch.cat((Z, invariant), dim=2)
        X_fake = self.decoder(Z, y)
        return X_fake, mu, sigma, invariant


if __name__ == '__main__':
    X = torch.randn((16, 10, 258, 25), dtype=torch.float32, device='cpu', requires_grad=False)
    y = torch.randint(0, 5, (16, 10), dtype=torch.int64, device='cpu', requires_grad=False)
    z = torch.randn((16, 10, 128), dtype=torch.float32, device='cpu', requires_grad=False)
    encoder = Encoder(2)
    decoder = Decoder(2)
    mu, sigma, invariant = encoder(X, y)
    Z = mu + z * sigma.exp()
    Z = torch.cat((Z, invariant), dim=2)
    X_fake = decoder(Z, y)
    print(X_fake.shape)
    torch.save(encoder.state_dict(), 'cvae_encoder.pth')
    torch.save(decoder.state_dict(), 'cvae_decoder.pth')
    cvae = CVAE(2)
    X_fake, mu, sigma, invariant = cvae(X, y, z)
    print(X_fake.shape, mu.shape, sigma.shape)
    torch.save(cvae.state_dict(), 'cvae_network.pth')
