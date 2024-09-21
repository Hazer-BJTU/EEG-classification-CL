from clnetworks import *
from torch.utils.data import DataLoader


def train_cl(args, trains, valids, tests):
    test_results = []
    clnetwork = CLnetwork(args)
    for task_idx in range(args.task_num):
        print(f'start task {task_idx}:')
        clnetwork.start_task()
        train_loader = DataLoader(trains[task_idx], args.batch_size, True)
        for epoch in range(args.num_epochs):
            clnetwork.start_epoch()
            for X, y in train_loader:
                clnetwork.observe(X, y)
            clnetwork.end_epoch(valids[task_idx])
        clnetwork.end_task()
        confusion = ConfusionMatrix(args.task_num)
        confusion = evaluate_tasks(clnetwork.best_net_memory[task_idx], tests, confusion, clnetwork.device, args.valid_batch)
        test_results.append(confusion.accuracy())
    print(test_results)


if __name__ == '__main__':
    pass
