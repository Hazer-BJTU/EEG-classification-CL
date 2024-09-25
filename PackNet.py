import torch
from clnetworks import *
import numpy as np
from main import args


class PackNetCLnetwork(CLnetwork):
    def __init__(self, args):
        super(PackNetCLnetwork, self).__init__(args)
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
        self.fixed = torch.ones(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        self.using = torch.ones(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        self.proportion = self.grad_number // args.task_num + 1
        self.fixed_numbers = 0
        self.start_fine_tuning = False

    def mask_gradient(self):
        idx = 0
        for param in self.net.parameters():
            if param.grad is not None:
                starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
                target = self.fixed[starting:ending].view(param.grad.shape)
                param.grad *= target
                idx += 1

    def mask_params(self):
        idx = 0
        with torch.no_grad():
            for param in self.net.parameters():
                if param.grad is not None:
                    starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
                    target = self.using[starting:ending].view(param.data.shape)
                    param.data *= target
                    idx += 1

    def get_params(self):
        idx = 0
        parameters = torch.zeros(self.grad_number, dtype=torch.float32, requires_grad=False, device=self.device)
        for param in self.net.parameters():
            if param.grad is not None:
                starting, ending = self.grad_positions[idx][0], self.grad_positions[idx][1]
                parameters[starting:ending].copy_(param.data.view(-1))
                idx += 1
        return parameters

    def sort_params(self):
        parameters = self.get_params()
        parameters_left = self.grad_number - self.fixed_numbers
        print(f'flexible parameters left: {parameters_left}')
        self.using.copy_(1 - self.fixed)
        if parameters_left <= self.proportion:
            self.using.fill_(1)
        else:
            params_sorted, idx_lst = torch.sort(parameters, descending=True)
            self.using.index_fill_(0, idx_lst[self.fixed_numbers:self.fixed_numbers + self.proportion], 1)
        print('parameter masks updated.')

    def fix_params(self):
        self.fixed_numbers = int(torch.sum(self.using).item())
        self.fixed.copy_(1 - self.using)
        print('parameters fixed.')

    def start_task(self):
        self.epoch = 0
        self.best_net = copy.deepcopy(self.net)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.args.num_epochs // 6, 1), 0.6)
        self.start_fine_tuning = False

    def start_epoch(self):
        super(PackNetCLnetwork, self).start_epoch()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.epoch >= self.args.num_epochs // 2 and not self.start_fine_tuning:
            self.start_fine_tuning = True
            print('start fine-tuning...')
            self.sort_params()
        if self.start_fine_tuning:
            self.mask_params()
        y_hat = self.net(X)
        L_current = torch.sum(self.loss(y_hat, y.view(-1)))
        L = L_current / X.shape[0]
        L.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.mask_gradient()
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        super(PackNetCLnetwork, self).end_epoch(valid_dataset)

    def end_task(self):
        super(PackNetCLnetwork, self).end_task()
        self.fix_params()


if __name__ == '__main__':
    clnetwork = PackNetCLnetwork(args)
    clnetwork.start_task()
    clnetwork.sort_params()
    clnetwork.fix_params()
    clnetwork.sort_params()
    clnetwork.fix_params()
    clnetwork.sort_params()
    clnetwork.fix_params()
    clnetwork.sort_params()
    clnetwork.fix_params()
    clnetwork.sort_params()
