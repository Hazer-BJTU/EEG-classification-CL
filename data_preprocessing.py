import numpy as np
import scipy.io as sio
import torch
from scipy import signal
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle as pkl


def load_data_isruc1(filepath, window_size, channels, total_num):
    file_names = [file for file in os.listdir(filepath) if file.endswith('.mat')]
    file_names.sort()
    datas, labels = [], []
    for file in file_names:
        raw_data = sio.loadmat(os.path.join(filepath, file))
        print(f'loading raw data from {os.path.join(filepath, file)}...')
        X = None
        for channel in channels:
            data_resampled = signal.resample(raw_data[channel], 3000, axis=1)
            print(f'calculating stft for channel {channel} in isruc1...')
            _, _, Zxx = signal.stft(data_resampled, 200, 'hann', 256)
            Zxx = 20 * np.log10(np.abs(Zxx))
            if X is None:
                X = torch.tensor(Zxx, dtype=torch.float32, requires_grad=False)
            else:
                temp = torch.tensor(Zxx, dtype=torch.float32, requires_grad=False)
                X = torch.cat((X, temp), dim=1)
        label_name = file.split('.')[0][7:] + '_1.npy'
        label = np.load(os.path.join(filepath, 'label', label_name))
        y = torch.tensor(label, dtype=torch.int64, requires_grad=False)
        data_seq, label_seq, segs = [], [], X.shape[0] // window_size
        for idx in range(segs):
            data_seq.append(X[idx * window_size: (idx + 1) * window_size])
            label_seq.append(y[idx * window_size: (idx + 1) * window_size])
        datas.append(data_seq)
        labels.append(label_seq)
        if len(datas) >= total_num:
            break
    return datas, labels


def load_data_shhs(filepath, window_size, channels, total_num):
    file_names = [file for file in os.listdir(filepath) if file.endswith('.pkl')]
    file_names.sort()
    shhs_channels = ['EEG', "EEG(sec)", 'EOG(L)', 'EMG']
    channel_index = [shhs_channels.index(c) for c in channels]
    datas, labels = [], []
    for file in file_names:
        raw_data = None
        with open(os.path.join(filepath, file), 'rb') as data_file:
            raw_data = pkl.load(data_file)
        print(raw_data['new_xall'])


class DataWrapper(Dataset):
    def __init__(self, data, labels, task):
        assert len(data) == len(labels) == len(task)
        self.data = data
        self.labels = labels
        self.task = task

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.task[item]

    def __len__(self):
        return len(self.data)


def create_fold(train, valid, test, datas_tasklist, labels_tasklist):
    train_data, train_label, train_task = [], [], []
    valid_data, valid_label, valid_task = [], [], []
    test_data, test_label, test_task = [], [], []
    t = 0
    for datas, labels in zip(datas_tasklist, labels_tasklist):
        for idx in train:
            for X, y in zip(datas[idx], labels[idx]):
                train_data.append(X)
                train_label.append(y)
                train_task.append(t)
        for idx in valid:
            for X, y in zip(datas[idx], labels[idx]):
                valid_data.append(X)
                valid_label.append(y)
                valid_task.append(t)
        for idx in test:
            for X, y in zip(datas[idx], labels[idx]):
                test_data.append(X)
                test_label.append(y)
                test_task.append(t)
        t += 1
    train_dataset = DataWrapper(train_data, train_label, train_task)
    valid_dataset = DataWrapper(valid_data, valid_label, valid_task)
    test_dataset = DataWrapper(test_data, test_label, test_task)
    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    '''
    datas, labels = load_data_isruc1('/home/ShareData/ISRUC-1/ISRUC-1', 10, ['F3_A2', 'ROC_A1'], 5)
    train, valid, test = create_fold([0, 1, 2], [3], [4], [datas, datas, datas], [labels, labels, labels])
    train_loader = DataLoader(train, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid, batch_size=8, shuffle=False)
    test_loader = DataLoader(test, batch_size=8, shuffle=False)
    print('train loader...')
    for X, y, t in train_loader:
        print(f'{X.shape}, {y.shape}, {t}')
    print('valid loader...')
    for X, y, t in valid_loader:
        print(f'{X.shape}, {y.shape}, {t}')
    print('test loader...')
    for X, y, t in test_loader:
        print(f'{X.shape}, {y.shape}, {t}')
    '''
    load_data_shhs('/home/ShareData/shhs1_process6', 10, ['EEG', 'EOG(L)'], 5)
