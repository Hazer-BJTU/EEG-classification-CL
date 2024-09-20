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
    parser.add_argument('--shhs', nargs='+', default=['EEG', 'EOG(L)'])
    args = parser.parse_args()
    datas1, labels1 = load_data_isruc1(args.isruc1_path, args.window_size, args.isruc1, args.total_num)
    datas2, labels2 = load_data_shhs(args.shhs_path, args.window_size, args.shhs, args.total_num)
    datas = [datas1, datas2]
    labels = [labels1, labels2]
    train, valid, test = create_fold([0, 1, 2], [3], [4], datas, labels)
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
