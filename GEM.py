import torch
from clnetworks import *
import quadprog
import numpy as np
from main import args


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

    def clac_old_gradient(self):
        self.G = None
        print(f'calculating old gradients on {len(self.memory_buffer)} tasks...')
        for sample in self.memory_buffer:
            self.optimizer.zero_grad()
            X_replay, y_replay = sample[0].to(self.device), sample[1].to(self.device)
            y_hat_replay = self.net(X_replay)
            L_replay = torch.sum(self.loss(y_hat_replay, y_replay.view(-1)))
            L_replay.backward()
            if self.G is None:
                self.G = torch.unsqueeze(self.get_gradient(), dim=0)
            else:
                gk = torch.unsqueeze(self.get_gradient(), dim=0)
                self.G = torch.cat((self.G, gk), dim=0)

    def gradient_projection(self, g):
        H = (self.G / self.grad_number) @ self.G.T
        H = H.cpu().numpy().astype(np.double)
        f = (self.G / self.grad_number) @ torch.unsqueeze(g, dim=1)
        f = torch.squeeze(f, dim=1).cpu().numpy().astype(np.double)
        a = np.eye(H.shape[0])
        b = np.zeros(H.shape[0])
        v = quadprog.solve_qp(H, -f, a, b)[0]
        v = torch.tensor(v.astype(np.float32), dtype=torch.float32, requires_grad=False, device=self.device)
        g_projected = self.G.T @ torch.unsqueeze(v, dim=1)
        g_projected = torch.squeeze(g_projected, dim=1) + g
        return g_projected


if __name__ == '__main__':
    clnetworks = GEMCLnetwork(args)
    for i in range(3):
        clnetworks.memory_buffer.append((torch.randn(128, 10, 258, 25), torch.zeros(128, 10, dtype=torch.int64)))
    clnetworks.clac_old_gradient()
    clnetworks.gradient_projection(clnetworks.get_gradient())
