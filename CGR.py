import torch
import math
import copy
from clnetworks import *
from generator import *
from models import init_weight
from torchvision import transforms
unloader = transforms.ToPILImage()


class CGRnetwork(NaiveCLnetwork):
    def __init__(self, args):
        super(CGRnetwork, self).__init__(args)
        self.generator_memories = []
        self.cvae = CVAE(args.channels_num)
        self.cvae.apply(init_weight)
        self.cvae.to(self.device)
        self.optimizerG = torch.optim.Adam(self.cvae.parameters(), lr=args.generator_lr)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, max(self.args.num_epochs // 6, 1), 0.6)
        self.cvae_loss = [0, 0, 0]
        self.running_mean, self.running_mean_sqr = 0, 0
        self.running_memory = []
        self.generator_memories = []

    def start_task(self):
        super(CGRnetwork, self).start_task()
        self.cvae.apply(init_weight)
        self.optimizerG = torch.optim.Adam(self.cvae.parameters(), lr=self.args.generator_lr)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, max(self.args.num_epochs // 6, 1), 0.6)
        self.running_mean, self.running_mean_sqr = 0, 0

    def start_epoch(self):
        super(CGRnetwork, self).start_epoch()
        self.cvae_loss = [0, 0, 0]

    def reservoir_sampling(self, X, y):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        alpha = self.observed_samples / (self.observed_samples + batch_size)
        beta = batch_size / (self.observed_samples + batch_size)
        '''calculate avg(X) and avg(X^2)'''
        mean = torch.mean(X.view(-1))
        mean_sqr = torch.mean(X.pow(2).view(-1))
        '''update running avg(X) and running avg(X^2)'''
        self.running_mean = alpha * self.running_mean + beta * mean.item()
        self.running_mean_sqr = alpha * self.running_mean_sqr + beta * mean_sqr.item()
        for idx in range(X.shape[0]):
            self.observed_samples += 1
            if self.data_buffer is None:
                self.data_buffer = torch.unsqueeze(X[idx].clone(), dim=0)
                self.label_buffer = torch.unsqueeze(y[idx].clone(), dim=0)
                self.buffer_counter += 1
                continue
            if self.buffer_counter < self.buffer_size:
                data = torch.unsqueeze(X[idx].clone(), dim=0)
                label = torch.unsqueeze(y[idx].clone(), dim=0)
                self.data_buffer = torch.cat((self.data_buffer, data), dim=0)
                self.label_buffer = torch.cat((self.label_buffer, label), dim=0)
                self.buffer_counter += 1
            elif random.random() <= self.buffer_size / self.observed_samples:
                target = random.randint(0, self.buffer_size - 1)
                self.data_buffer[target].copy_(X[idx])
                self.label_buffer[target].copy_(y[idx])

    def observe(self, X, y, first_time=False):
        if first_time:
            self.reservoir_sampling(X, y)
            print(f'sampling {self.buffer_counter} examples, labels only...')
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = torch.sum(self.loss(y_hat, y.view(-1)))
        L = L_current / X.shape[0]
        '''start generative replay'''
        replay_number = 0
        for sample in self.memory_buffer:
            replay_number += sample[1].shape[0]
        print(f'generative replay on {len(self.memory_buffer)} tasks and {replay_number} examples...')
        idx = 0
        for sample, gmodel in zip(self.memory_buffer, self.generator_memories):
            y_replay = sample[1].to(self.device)
            z = torch.randn((y_replay.shape[0], y_replay.shape[1], 384),
                            dtype=torch.float32, requires_grad=False, device=self.device)
            X_replay = gmodel(z, y_replay)
            X_replay = X_replay * self.running_memory[idx][1] + self.running_memory[idx][0]
            y_hat_replay = self.net(X_replay)
            L_replay = torch.mean(self.loss(y_hat_replay, y_replay.view(-1)))
            L = L + L_replay
            idx += 1
        L.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)
        '''start training generator'''
        mean = self.running_mean
        var = (self.running_mean_sqr - self.running_mean ** 2) * (self.observed_samples / (self.observed_samples - 1))
        var = math.sqrt(var) + 1e-5
        self.cvae.train()
        z = torch.randn((X.shape[0], X.shape[1], 128), dtype=torch.float32, requires_grad=False, device=self.device)
        self.optimizerG.zero_grad()
        self.optimizer.zero_grad()
        X_fake, mu, sigma = self.cvae(X, y, z)
        y_g, y_r = self.net(X_fake * var + mean), self.net(X)
        L_R = self.args.cvae_coefs[0] * torch.nn.functional.mse_loss(X_fake, (X - mean) / var)
        L_KL = self.args.cvae_coefs[1] * torch.mean(0.5 * (mu.pow(2) + sigma.exp() - sigma - 1))
        L_N = self.args.cvae_coefs[2] * torch.mean(self.loss(y_g, y_r.softmax(dim=1)))
        self.cvae_loss[0] += L_R.item()
        self.cvae_loss[1] += L_KL.item()
        self.cvae_loss[2] += L_N.item()
        (L_R + L_N).backward()
        nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=20, norm_type=2)
        self.optimizerG.step()

    def end_epoch(self, valid_dataset):
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {self.optimizer.state_dict()['param_groups'][0]['lr'] * 1000:.3f}, "
              f'generator loss: {self.cvae_loss[0]:.3f} + {self.cvae_loss[1]:.3f} + {self.cvae_loss[2]:.3f}')
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks(self.net, [valid_dataset], valid_confusion,
                                             self.device, self.args.valid_batch)
            valid_acc, valid_mf1 = valid_confusion.accuracy(), valid_confusion.macro_f1()
            print(f'valid accuracy: {valid_acc:.3f}, valid macro F1: {valid_mf1:.3f}')
            if valid_acc > self.best_valid_acc:
                self.best_train_loss = self.train_loss
                self.best_train_acc = train_acc
                self.best_valid_acc = valid_acc
                self.best_net = './modelsaved/' + str(self.args.replay_mode) + '_task' + str(self.task) + '.pth'
                torch.save(self.net.state_dict(), self.best_net)
            if self.args.visualize:
                mean = self.running_mean
                var = (self.running_mean_sqr - self.running_mean ** 2) * (
                        self.observed_samples / (self.observed_samples - 1))
                var = math.sqrt(var)
                datas, labels = self.data_buffer.to(self.device), self.label_buffer.to(self.device)
                z = torch.randn((datas.shape[0], datas.shape[1], 128),
                                dtype=torch.float32, requires_grad=False, device=self.device)
                noise = torch.randn((datas.shape[0], datas.shape[1], 384),
                                    dtype=torch.float32, requires_grad=False, device=self.device)
                X_fake, _, _ = self.cvae(datas, labels, z)
                X_fake = torch.abs(X_fake * var + mean - datas).tanh()
                X_noise = self.cvae.decoder(noise, labels)
                X_noise = torch.abs(X_noise * var + mean - datas).tanh()
                for idx in range(datas.shape[1]):
                    image = X_fake[0][idx].clone()
                    image = unloader(image)
                    image.save(f'./visual/real_fake_diff_{idx}.jpg')
                    image = X_noise[0][idx].clone()
                    image = unloader(image)
                    image.save(f'./visual/real_noise_diff_{idx}.jpg')
        self.epoch += 1
        self.scheduler.step()
        self.schedulerG.step()

    def end_task(self):
        self.task += 1
        self.best_net_memory.append(self.best_net)
        self.memory_buffer.append((None, self.label_buffer))
        mean = self.running_mean
        var = (self.running_mean_sqr - self.running_mean ** 2) * (self.observed_samples / (self.observed_samples - 1))
        var = math.sqrt(var)
        self.running_memory.append((mean, var))
        self.generator_memories.append(copy.deepcopy(self.cvae.decoder))


if __name__ == '__main__':
    pass
