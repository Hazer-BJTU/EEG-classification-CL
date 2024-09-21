import torch
from models import SeqSleepNet
from torch.utils.data import DataLoader


class ConfusionMatrix:
    def __init__(self, num_tasks, num_catagories=5):
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


if __name__ == '__main__':
    pass
