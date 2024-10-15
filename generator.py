import torch
import torch.nn as nn
from models import *


class Label2Vec(nn.Module):
    def __init__(self, features, **kwargs):
        super(Label2Vec, self).__init__(**kwargs)
        self.features = features
        self.W = torch.randn((5, features), dtype=torch.float32, requires_grad=False)
        for i in range(self.W.shape[0]):
            alpha = self.W[i].clone()
            for j in range(0, i):
                k = torch.dot(alpha, self.W[j]) / torch.dot(self.W[j], self.W[j])
                self.W[i] -= k * self.W[j]
        self.W = torch.nn.functional.normalize(self.W, p=2, dim=1)

    def forward(self, X):
        X_one_hot = torch.nn.functional.one_hot(X, 5)
        X_one_hot = X_one_hot.to(torch.float32)
        self.W.to(X_one_hot.device)
        output = X_one_hot @ self.W
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
                self.block.add_module(f'module_#{idx}_bn', nn.BatchNorm1d(channels_lst[idx + 1]))

    def forward(self, X):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, F, T)
        X = self.block(X)
        X = X.view(batch_size, seq_length, -1, T)
        return X


class LongTermRNN(nn.Module):
    def __init__(self, input_size, hiddens, bidirectional=False, **kwargs):
        super(LongTermRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.block = nn.LSTM(input_size, hiddens, batch_first=True, bidirectional=bidirectional)
        self.d = 2 if bidirectional else 1

    def get_initial_states(self, batch_size, device):
        H0 = torch.zeros((self.d, batch_size, self.hiddens), device=device)
        C0 = torch.zeros((self.d, batch_size, self.hiddens), device=device)
        return H0, C0

    def forward(self, X):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        H0, C0 = self.get_initial_states(batch_size, X.device)
        output, (Hn, Cn) = self.block(X, (H0, C0))
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class Attention(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.input_size = input_size
        self.Watt = nn.Parameter(torch.randn((input_size, 1)))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X):
        A = torch.matmul(X, self.Watt)
        A = A.transpose(1, 2)
        A = self.softmax(A)
        X = torch.bmm(A, X)
        X = torch.squeeze(X, dim=1)
        return X


class Generator(nn.Module):
    def __init__(self, channels_num=2, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.label2vec = Label2Vec(128)
        self.rnn = LongTermRNN(256, 384)
        self.linear = nn.Sequential(nn.Linear(384, channels_num * 129), nn.ReLU())
        self.cnn = CNNlayer((1, 64, 128, 128, 64, 25), (17, 17, 17, 17, 17))
        self.tanh = nn.Tanh()

    def forward(self, Z, y):
        batch_size, seq_length = Z.shape[0], Z.shape[1]
        y = self.label2vec(y)
        Z = torch.cat((Z, y), dim=2)
        Z = self.rnn(Z)
        Z = Z.view(batch_size * seq_length, -1)
        Z = self.linear(Z)
        Z = torch.unsqueeze(Z.view(batch_size, seq_length, -1), dim=2)
        Z = self.cnn(Z)
        Z = self.tanh(Z)
        Z = Z.transpose(2, 3)
        return Z


class Discriminator(nn.Module):
    def __init__(self, channels_num=2, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.label2vec = Label2Vec(200)
        self.cnn = CNNlayer((channels_num * 129, 128, 64, 32, 16, 8), (5, 5, 5, 5, 5))
        self.rnn = LongTermRNN(400, 256)
        self.attention = Attention(256)
        self.linear = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128 ,64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X, y):
        batch_size, seq_length = X.shape[0], X.shape[1]
        y = self.label2vec(y)
        X = self.cnn(X)
        X = X.view(batch_size, seq_length, -1)
        X = torch.cat((X, y), dim=2)
        X = self.rnn(X)
        X = self.attention(X)
        X = self.linear(X)
        return X


if __name__ == '__main__':
    z = torch.randn((16, 10, 128), dtype=torch.float32, requires_grad=False, device='cpu')
    y = torch.randint(0, 5, (16, 10), dtype=torch.int64, requires_grad=False, device='cpu')
    netG, netD = Generator(2), Discriminator(2)
    X_fake = netG(z, y)
    print(X_fake.shape)
    pred = netD(X_fake, y)
    print(pred.shape)
    torch.save(netG.state_dict(), 'gnerator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
