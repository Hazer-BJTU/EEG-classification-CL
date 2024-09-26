import copy
import torch
from models import SeqSleepNet
from torch.utils.data import DataLoader


class ConfusionMatrix:
    def __init__(self, num_tasks, num_catagories=5):
        self.num_tasks = num_tasks
        self.num_catagories = num_catagories
        self.mat = torch.zeros((num_tasks, num_catagories, num_catagories), dtype=torch.int64, requires_grad=False)

    def count(self, y_hat, y, t):
        y_hat = torch.argmax(y_hat, dim=1)
        y, t = y.view(-1), t.view(-1)
        window_size = y.shape[0] // t.shape[0]
        for idx in range(y.shape[0]):
            self.mat[t[idx // window_size]][y_hat[idx]][y[idx]] += 1

    def count_task_separated(self, y_hat, y, t):
        y_hat = torch.argmax(y_hat, dim=1)
        y = y.view(-1)
        for idx in range(y.shape[0]):
            self.mat[t][y_hat[idx]][y[idx]] += 1

    def accuracy(self):
        acc = []
        for idx in range(self.num_tasks):
            total = torch.sum(self.mat[idx]).item()
            true = 0
            for i in range(self.num_catagories):
                true += self.mat[idx][i][i].item()
            acc.append(true / max(total, 1))
        if len(acc) == 1:
            return acc[0]
        else:
            return acc

    def macro_f1(self):
        mf1 = []
        for idx in range(self.num_tasks):
            f1 = 0
            for i in range(self.num_catagories):
                row, column = 0, 0
                for j in range(self.num_catagories):
                    row += self.mat[idx][i][j].item()
                    column += self.mat[idx][j][i].item()
                precision = self.mat[idx][i][i].item() / max(row, 1)
                recall = self.mat[idx][i][i].item() / max(column, 1)
                f1 += 2 * precision * recall / max(precision + recall, 1)
            f1 /= self.num_catagories
            mf1.append(f1)
        if len(mf1) == 1:
            return mf1[0]
        else:
            return mf1

    def clear(self):
        self.mat.zero_()

    def __getitem__(self, item):
        return self.mat[item]


def evaluate(net, loader, confusion_matrix, device):
    net.eval()
    with torch.no_grad():
        for X, y, t in loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            confusion_matrix.count(y_hat, y, t)
    return confusion_matrix


def evaluate_tasks(net, datasets, confusion_matrix, device, batch_size=1):
    net.eval()
    with torch.no_grad():
        for idx in range(len(datasets)):
            loader = DataLoader(datasets[idx], batch_size=batch_size, shuffle=False)
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                confusion_matrix.count_task_separated(y_hat, y, idx)
    return confusion_matrix


def evaluate_tasks_packnet(net, datasets, confusion_matrix, device, clnetwork, batch_size=1):
    using_list = clnetwork.using_list
    grad_positions = clnetwork.grad_positions
    with torch.no_grad():
        for idx in range(len(datasets)):
            masked_net = copy.deepcopy(net)
            if idx < len(using_list):
                print(f'start masking parameters, testing on dataset {idx}, '
                      f'parameters size {int(torch.sum(using_list[idx]).item())}')
                cnt = 0
                for param in masked_net.parameters():
                    starting, ending = grad_positions[cnt][0], grad_positions[cnt][1]
                    target = using_list[idx][starting:ending].view(param.data.shape)
                    param.data *= target
                    cnt += 1
            else:
                print(f'start testing without masking parameters on dataset {idx}...')
            masked_net.eval()
            loader = DataLoader(datasets[idx], batch_size=batch_size, shuffle=False)
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                y_hat = masked_net(X)
                confusion_matrix.count_task_separated(y_hat, y, idx)
    return confusion_matrix


if __name__ == '__main__':
    pass
