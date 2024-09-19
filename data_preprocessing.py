import numpy as np
import scipy.io as sio
import torch
from scipy import signal
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
                X = torch.unsqueeze(torch.tensor(Zxx, dtype=torch.float32, requires_grad=False), dim=1)
            else:
                temp = torch.unsqueeze(torch.tensor(Zxx, dtype=torch.float32, requires_grad=False), dim=1)
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


class DataWrapper(Dataset):
    def __init__(self, data, labels, task):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
        self.task = [task for _ in range(len(data))]

    def __getitem__(self, item):
        return self.data[item], self.labels[item], self.task[item]

    def __len__(self):
        return len(self.data)


def create_fold(train, valid, test, datas, labels, task):
    train_datas = [item for idx in train for item in datas[idx]]
    train_labels = [item for idx in train for item in labels[idx]]
    valid_datas = [item for idx in valid for item in datas[idx]]
    valid_labels = [item for idx in valid for item in labels[idx]]
    test_datas = [item for idx in test for item in datas[idx]]
    test_labels = [item for idx in test for item in labels[idx]]
    trainset = DataWrapper(train_datas, train_labels, task)
    validset = DataWrapper(valid_datas, valid_labels, task)
    testset = DataWrapper(test_datas, test_labels, task)
    return trainset, validset, testset


if __name__ == '__main__':
    datas, labels = load_data_isruc1('/home/ShareData/ISRUC-1/ISRUC-1', 10, ['F3_A2', 'ROC_A1'], 5)
    train, valid, test = create_fold([0, 1, 2], [3], [4], datas, labels, 0)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    for X, y, t in train_loader:
        print(f'{X.shape}, {y.shape}, {t}')
