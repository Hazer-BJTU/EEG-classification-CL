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


class CNNlayer(nn.Module):
    def __init__(self, channels_lst, kernels_lst, **kwargs):
        super(CNNlayer, self).__init__(**kwargs)
        self.channels_lst = channels_lst
        self.kernels_lst = kernels_lst
        assert len(channels_lst) > 1
        self.block = nn.Sequential()
        for idx in range(0, len(channels_lst) - 1):
            self.block.add_module(f'module_#{idx}_conv',
                                  nn.Conv1d(channels_lst[idx], channels_lst[idx + 1],
                                            kernel_size=kernels_lst[idx], stride=1, padding='same'))
            if idx != len(channels_lst) - 2:
                self.block.add_module(f'module_#{idx}_relu', nn.ReLU())

    def forward(self, X):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, F, T)
        X = self.block(X)
        X = X.view(batch_size, seq_length, -1, T)
        return X


class ShortTermGRU(nn.Module):
    def __init__(self, input_size, hiddens, **kwargs):
        super(ShortTermGRU, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.block = nn.GRU(input_size, hiddens, batch_first=True, bidirectional=False)

    def get_initial_states(self, batch_size, device):
        return torch.zeros((1, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.transpose(2, 3).view(batch_size * seq_length, T, F)
        H0 = self.get_initial_states(X.shape[0], X.device)
        (output, Hn) = self.block(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        output = output.view(batch_size, seq_length, T, self.hiddens)
        return output


class LongTermGRU(nn.Module):
    def __init__(self, input_size, hiddens, **kwargs):
        super(LongTermGRU, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.block = nn.GRU(input_size, hiddens, batch_first=True, bidirectional=False)

    def get_initial_states(self, batch_size, device):
        return torch.zeros((1, batch_size, self.hiddens), device=device)

    def forward(self, X):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        H0 = self.get_initial_states(batch_size, X.device)
        (output, Hn) = self.block(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class Resblock(nn.Module):
    def __init__(self, input_features, output_features, **kwargs):
        super(Resblock, self).__init__(**kwargs)
        self.input_features = input_features
        self.output_features = output_features
        self.block = nn.Sequential(nn.Linear(input_features, output_features), nn.ReLU())

    def forward(self, X):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.transpose(2, 3).contiguous().view(batch_size * seq_length * T, F)
        X = self.block(X)
        X = X.view(batch_size, seq_length, T, self.output_features)
        return X


class Gnerator(nn.Module):
    def __init__(self, channels_num=2, **kwargs):
        super(Gnerator, self).__init__(**kwargs)
        self.channels_num = channels_num
        self.label2vec = Label2Vec(400)
        self.cnn = CNNlayer((channels_num * 129, 128, 64, 32, 16), (3, 3, 3, 3))
        self.long_term_gru = LongTermGRU(800, 256)
        self.short_term_gru = ShortTermGRU(channels_num * 129, 256)
        self.linear = nn.Sequential(
            nn.Linear(512, 768), nn.ReLU(),
            nn.Linear(768, channels_num * 129), nn.Tanh()
        )

    def forward(self, X, y):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        y = self.label2vec(y)
        ld = self.cnn(X)
        ld = ld.view(batch_size, seq_length, -1)
        ld = torch.cat((ld, y), dim=2)
        ld = self.long_term_gru(ld)
        ld = torch.unsqueeze(ld, dim=2)
        ld = ld.expand(batch_size, seq_length, T, ld.shape[3])
        sd = self.short_term_gru(X)
        Z = torch.cat((ld, sd), dim=3)
        Z = Z.view(batch_size * seq_length * T, -1)
        output = self.linear(Z)
        output = output.view(batch_size, seq_length, T, -1).transpose(2, 3)
        return output


class Discriminator(nn.Module):
    def __init__(self, channels_num=2, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.channels_num = channels_num
        self.label2vec = Label2Vec(400)
        self.cnn = CNNlayer((channels_num * 129, 128, 64, 32, 16), (3, 3, 3, 3))
        self.long_term_gru = LongTermGRU(800, 256)
        self.linear = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, X, y):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        y = self.label2vec(y)
        ld = self.cnn(X)
        ld = ld.view(batch_size, seq_length, -1)
        ld = torch.cat((ld, y), dim=2)
        ld = self.long_term_gru(ld)
        ld = ld.view(batch_size * seq_length, -1)
        output = self.linear(ld)
        output = output.view(batch_size, seq_length)
        return output


if __name__ == '__main__':
    y = torch.randint(0, 5, (16, 10), dtype=torch.int64, device='cpu', requires_grad=False)
    z = torch.randn((16, 10, 258, 25), dtype=torch.float32, device='cpu', requires_grad=False)
    netG, netD = Gnerator(), Discriminator()
    X = netG(z, y)
    print(X.shape)
    pred = netD(X, y)
    print(pred.shape)
    torch.save(netG.state_dict(), 'gnerator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
