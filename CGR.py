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
        self.running_mean = torch.zeros((args.channels_num * 129, 25),
                                        dtype=torch.float32, requires_grad=False, device=self.device)
        self.running_mean_sqr = torch.zeros((args.channels_num * 129, 25),
                                            dtype=torch.float32, requires_grad=False, device=self.device)
        self.running_memory = []
        self.generator = Gnerator(args.channels_num)
        self.discriminator = Discriminator(args.channels_num)
        self.optimizerG, self.optimizerD = None, None
        self.schedulerG, self.schedulerD = None, None
        self.bceloss = nn.BCEWithLogitsLoss()
        self.generative_loss = [0, 0, 0]
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator_memory = []

    def start_task(self):
        super(CGRnetwork, self).start_task()
        self.running_mean = torch.zeros((self.args.channels_num * 129, 25),
                                        dtype=torch.float32, requires_grad=False, device=self.device)
        self.running_mean_sqr = torch.zeros((self.args.channels_num * 129, 25),
                                            dtype=torch.float32, requires_grad=False, device=self.device)
        self.generator.apply(init_weight)
        self.discriminator.apply(init_weight)
        self.optimizerG = torch.optim.SGD(self.generator.parameters(), lr=self.args.generator_lr)
        self.optimizerD = torch.optim.SGD(self.discriminator.parameters(), lr=self.args.generator_lr)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, max(self.args.num_epochs // 6, 1), 0.6)
        self.schedulerD = torch.optim.lr_scheduler.StepLR(self.optimizerD, max(self.args.num_epochs // 6, 1), 0.6)

    def start_epoch(self):
        super(CGRnetwork, self).start_epoch()
        self.generative_loss = [0, 0, 0]

    def reservoir_sampling(self, X, y):
        batch_size, seq_length, F, T = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        alpha = self.observed_samples / (self.observed_samples + batch_size)
        beta = batch_size / (self.observed_samples + batch_size)
        '''calculate avg(X) and avg(X^2)'''
        mean = torch.mean(X.view(-1, X.shape[2], X.shape[3]), dim=0).to(self.device)
        mean_sqr = torch.mean(X.pow(2).view(-1, X.shape[2], X.shape[3]), dim=0).to(self.device)
        '''update running avg(X) and running avg(X^2)'''
        self.running_mean = alpha * self.running_mean + beta * mean
        self.running_mean_sqr = alpha * self.running_mean_sqr + beta * mean_sqr
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
        '''generative replay'''
        idx = 0
        for sample, model in zip(self.memory_buffer, self.generator_memory):
            model = model.to(self.device).eval()
            y_replay = sample[1].to(self.device)
            z = torch.randn((y_replay.shape[0], y_replay.shape[1], 64),
                            dtype=torch.float32, requires_grad=False, device=self.device)
            X_fake = model(z, y_replay)
            X_fake = X_fake.detach() * self.running_memory[idx][1] + self.running_memory[idx][0]
            y_replay_hat = self.net(X_fake)
            L_replay = torch.sum(self.loss(y_replay_hat, y_replay.view(-1)))
            L = L + L_replay / y_replay.shape[0]
            idx += 1
        L.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.train_loss += L.item()
        self.confusion_matrix.count_task_separated(y_hat, y, 0)
        '''train generator'''
        mean = self.running_mean
        var = (self.running_mean_sqr - self.running_mean.pow(2)) * (self.observed_samples / (self.observed_samples - 1))
        var = var.pow(0.5)
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()
        z = torch.randn((X.shape[0], X.shape[1], 64), dtype=torch.float32, requires_grad=False, device=self.device)
        target = torch.ones((X.shape[0], X.shape[1]), dtype=torch.float32, requires_grad=False, device=self.device)
        X_fake = self.generator(z, y)
        pred_d = self.discriminator(X_fake * var + mean, y)
        pred_n = self.net(X_fake * var + mean)
        L_G = self.bceloss(pred_d.view(-1), target.view(-1))
        L_N = torch.sum(self.loss(pred_n, y_hat.detach().softmax(dim=1))) / X.shape[0] * self.args.cgr_coef
        self.generative_loss[0] += L_G.item()
        (L_G + L_N).backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=20, norm_type=2)
        self.optimizerG.step()
        '''train discriminator'''
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()
        target = torch.randint(0, 2, (y.shape[0], y.shape[1]), dtype=torch.float32, device=self.device)
        choice = target.view(y.shape[0], y.shape[1], 1, 1).expand(X.shape)
        maskt, maskg = choice, 1 - choice
        noise = torch.randn(X.shape, dtype=torch.float32, requires_grad=False, device=self.device) * var * 0.33
        discriminator_input = maskt * X + maskg * (X_fake.detach() * var + mean) + noise
        pred = self.discriminator(discriminator_input, y)
        L_D = self.bceloss(pred.view(-1), target.view(-1))
        self.generative_loss[1] += L_D.item()
        L_D.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=20, norm_type=2)
        self.optimizerD.step()
        '''validation'''
        pred_n = self.net(X_fake.detach() * var + mean)
        L_N = torch.sum(self.loss(pred_n, y.view(-1))) / X.shape[0]
        self.generative_loss[2] += L_N.item()

    def end_epoch(self, valid_dataset):
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss:.3f}, train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {self.optimizer.state_dict()['param_groups'][0]['lr'] * 1000:.3f}, "
              f'generative loss: {self.generative_loss[0]:.3f}, {self.generative_loss[1]:.3f}, {self.generative_loss[2]:.3f}')
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
                var = (self.running_mean_sqr - self.running_mean.pow(2)) * (self.observed_samples / (self.observed_samples - 1))
                var = var.pow(0.5)
                datas, labels = self.data_buffer.to(self.device), self.label_buffer.to(self.device)
                z = torch.randn((datas.shape[0], datas.shape[1], 64),
                                dtype=torch.float32, requires_grad=False, device=self.device)
                datas_fake = self.generator(z, labels)
                datas_fake = datas_fake * var + mean
                for idx in range(datas.shape[1]):
                    image1 = datas[0][idx].detach()
                    image2 = datas_fake[0][idx].detach()
                    minn = torch.min(image1).item()
                    maxn = torch.max(image1 - minn).item()
                    dvdline = torch.ones((image1.shape[0], 1), dtype=image1.dtype, device=image1.device) * 2
                    image = torch.cat(((image1 - minn) / maxn, dvdline, (image2 - minn) / maxn), dim=1)
                    image = unloader(image)
                    image.save(f'./visual/real_fake_example_{idx}.jpg')
        self.epoch += 1
        self.scheduler.step()
        self.schedulerG.step()
        self.schedulerD.step()

    def end_task(self):
        self.task += 1
        self.best_net_memory.append(self.best_net)
        self.memory_buffer.append((None, self.label_buffer))
        mean = self.running_mean
        var = (self.running_mean_sqr - self.running_mean.pow(2)) * (self.observed_samples / (self.observed_samples - 1))
        var = var.pow(0.5)
        self.running_memory.append((mean, var))
        self.generator_memory.append(copy.deepcopy(self.generator).to('cpu'))


if __name__ == '__main__':
    pass
