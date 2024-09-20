from data_preprocessing import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment settings')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0, help='device index')
    parser.add_argument('--window_size', type=int, nargs='?', default=10, help='length of sequence')
    parser.add_argument('--total_num', type=int, nargs='?', default=5, help='number of examples for each task')
    parser.add_argument('--isruc1_path', type=str, nargs='?',
                        default='/home/ShareData/ISRUC-1/ISRUC-1', help='file path of isruc1 dataset')
    parser.add_argument('--isruc1', nargs='+', default=['C4_A1', 'LOC_A2'], help='channels for isruc1')
    parser.add_argument('--shhs_path', type=str, nargs='?',
                        default='/home/ShareData/shhs1_process6', help='file path of shhs dataset')
    parser.add_argument('--shhs', nargs='+', default=['EEG', 'EOG(L)'], help='channels for shhs')
    parser.add_argument('--mass_path', type=str, nargs='?',
                        default='/home/ShareData/MASS_SS3_3000_25C-Cz', help='file path of mass dataset')
    parser.add_argument('--mass', nargs='+', default=['C4', 'EogL'], help='channels for mass')
    parser.add_argument('--sleep_edf_path', type=str, nargs='?',
                        default='/home/ShareData/sleep-edf-153-3chs', help='file path of sleepedf dataset')
    parser.add_argument('--sleep_edf', nargs='+', default=['Fpz-Cz', 'EOG'], help='channels of sleepedf')
    args = parser.parse_args()
    datas, labels = load_all_datasets(args)
    train, valid, test = create_fold([0, 1, 2], [3], [4], datas, labels)
    train_loader = DataLoader(train, batch_size=16, shuffle=False)
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
