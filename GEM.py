import torch
from clnetworks import *
import quadprog
import numpy as np


class GEMCLnetwork(NaiveCLnetwork):
    def __init__(self, args):
        super(GEMCLnetwork, self).__init__(args)
        test = torch.zeros(1, args.window_size, 258, 25).to(self.device)
        L = torch.sum(self.net(test))
        L.backward()
        self.grad_number = 0
        self.grad_positions = []
        for param in self.net.parameters():
            if param.grad is not None:
                grad_length = param.grad.view(-1).shape[0]
                self.grad_positions.append((self.grad_number, self.grad_number + grad_length))
                self.grad_number += grad_length
        self.G = None

    def get_gradient(self):
        grads = torch.zeros(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        idx = 0
        for param in self.net.parameters():
            if param.grad is not None:
                starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
                grads[starting:ending].copy_(param.grad.view(-1))
                idx += 1
        return grads

    def calc_old_gradient(self):
        self.G = None
        print(f'calculating old gradients on {len(self.memory_buffer)} tasks...')
        for sample in self.memory_buffer:
            self.optimizer.zero_grad()
            X_replay, y_replay = sample[0].to(self.device), sample[1].to(self.device)
            y_hat_replay = self.net(X_replay)
            L_replay = torch.sum(self.loss(y_hat_replay, y_replay.view(-1))) / X_replay.shape[0]
            L_replay.backward()
            if self.G is None:
                self.G = torch.unsqueeze(self.get_gradient(), dim=0)
            else:
                gk = torch.unsqueeze(self.get_gradient(), dim=0)
                self.G = torch.cat((self.G, gk), dim=0)

    def gradient_projection(self, g, eps=1e-3, margin=0.5, coef=3):
        factor = torch.norm(g).item() / (self.G.shape[0] + 1)
        for idx in range(self.G.shape[0]):
            factor += torch.norm(self.G[idx]).item() / (self.G.shape[0] + 1)
        H = (self.G / factor) @ (self.G.T / factor)
        H = H.cpu().numpy().astype(np.double)
        H = 0.5 * (H + H.transpose()) + np.eye(H.shape[0]) * eps
        f = (self.G / factor) @ (torch.unsqueeze(g, dim=1) / factor)
        f = torch.squeeze(f, dim=1).cpu().numpy().astype(np.double)
        a = np.eye(H.shape[0])
        b = np.zeros(H.shape[0]) + margin
        v = quadprog.solve_qp(H, -f, a, b)[0] * coef
        print(f'projection vector: {np.round(v, 3)}')
        v = torch.tensor(v.astype(np.float32), dtype=torch.float32, requires_grad=False, device=self.device)
        g_projected = self.G.T @ torch.unsqueeze(v, dim=1)
        g_projected = torch.squeeze(g_projected, dim=1) + g
        return g_projected

    def rewrite_gradient(self, g_projectd):
        idx = 0
        for param in self.net.parameters():
            if param.grad is not None:
                starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
                target = g_projectd[starting:ending].view(param.grad.shape)
                param.grad.copy_(target)
                idx += 1

    def start_task(self):
        super(GEMCLnetwork, self).start_task()

    def start_epoch(self):
        super(GEMCLnetwork, self).start_epoch()

    def observe(self, X, y, first_time=False):
        if first_time:
            self.reservoir_sampling(X, y)
            print(f'sampling {self.buffer_counter} examples...')
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = torch.sum(self.loss(y_hat, y.view(-1)))
        L = L_current / X.shape[0]
        L.backward()
        if self.task > 0:
            print(f'start gradient projection...')
            g = self.get_gradient()
            self.optimizer.zero_grad()
            self.calc_old_gradient()
            g_projected = self.gradient_projection(g)
            self.rewrite_gradient(g_projected)
            print(f'rewrite gradient finished.')
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        super(GEMCLnetwork, self).end_epoch(valid_dataset)

    def end_task(self):
        super(GEMCLnetwork, self).end_task()


if __name__ == '__main__':
    pass
