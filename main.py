from data_preprocessing import *
from train import *
from models import *
import argparse

parser = argparse.ArgumentParser(description='experiment settings')
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
parser.add_argument('--task_num', type=int, nargs='?', default=4, help='number of tasks')
parser.add_argument('--task_names', nargs='+', default=['ISRUC1', 'SHHS', 'MASS', 'SLEEP-EDF'],
                    help='the list of task names')
parser.add_argument('--cuda_idx', type=int, nargs='?', default=0, help='device index')
parser.add_argument('--window_size', type=int, nargs='?', default=10, help='length of sequence')
parser.add_argument('--total_num', type=int, nargs='?', default=60, help='number of examples for each task')
parser.add_argument('--fold_num', type=int, nargs='?', default=6, help='number of a single fold')
parser.add_argument('--num_epochs', type=int, nargs='?', default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, nargs='?', default=256, help='batch size')
parser.add_argument('--valid_epoch', type=int, nargs='?', default=10, help='validating interval')
parser.add_argument('--valid_batch', type=int, nargs='?', default=32, help='validating batch size')
parser.add_argument('--dropout', type=float, nargs='?', default=0.5, help='drop out value')
parser.add_argument('--weight_decay', type=float, nargs='?', default=0, help='weight decay value')
parser.add_argument('--lr', type=float, nargs='?', default=1e-3, help='learning rate')
parser.add_argument('--replay_mode', type=str, nargs='?', default='none', help='continual learning strategy')
parser.add_argument('--buffer_size', type=int, nargs='?', default=16, help='number of examples stored per task')
parser.add_argument('--channels_num', type=int, nargs='?', default=2, help='number of channels')
parser.add_argument('--generator_lr', type=float, nargs='?', default=1e-3, help='learning rate for GAN')
parser.add_argument('--visualize', type=bool, nargs='?', default=True, help='enable generative visualization')
parser.add_argument('--cgr_coef', type=float, nargs='?', default=1, help='coefficient for cgr L_N')
args = parser.parse_args()

if __name__ == '__main__':
    R = train_k_fold(args)
    write_format(R, args, 'cl_output_record_' + args.replay_mode + '.txt')
