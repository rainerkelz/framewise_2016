from torch import optim
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter
import datasets
import models

from utils import ChunkedRandomSampler
from utils import train
from utils import ensure_empty_directory_exists, filenames_from_splitfile
from utils import OneCyclePolicy
import os

import numpy as np

import argparse


def main():
    parser = argparse.ArgumentParser(description='train a VGG style conv net on the MAPS dataset')
    parser.add_argument('splits', type=str,
                        help='on which splits to train')
    parser.add_argument('run_path', type=str,
                        help='where to write run state')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    cuda = False
    if args.device.startswith('cpu'):
        cuda = False
    elif args.device.startswith('cuda'):
        cuda = True
    else:
        print('unknown device type "{}"'.format(args.device))
        exit(-1)

    run_path = args.run_path
    if os.path.exists(run_path):
        print('run_path "{}" already exists!'.format(run_path))
        exit(-1)

    ##########################################
    # prepare train data
    train_filenames = filenames_from_splitfile(os.path.join(args.splits, 'train'))
    train_sequences = datasets.get_sequences(train_filenames)

    batch_size = 32
    n_steps = 8192
    # one train loader for all train sequences
    train_dataset = ConcatDataset(train_sequences)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ChunkedRandomSampler(train_dataset, batch_size * n_steps),
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    # prepare valid data as individual sequences
    valid_filenames = filenames_from_splitfile(os.path.join(args.splits, 'valid'))

    # to get all of the sequence data, just write this:
    valid_sequences = datasets.get_sequences(valid_filenames)

    valid_loaders = []
    for valid_sequence in valid_sequences:
        valid_loader = DataLoader(
            valid_sequence,
            batch_size=1024,
            sampler=SequentialSampler(valid_sequence),
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        valid_loaders.append(valid_loader)

    log_dir = os.path.join(run_path, 'tensorboard')
    ensure_empty_directory_exists(log_dir)
    logger = SummaryWriter(log_dir=log_dir)

    net = models.VGGNet2016()

    # these would be the OneCyclePolicy learnrate
    # does not really improve things ...

    # n_ramp_up = n_steps // 2
    # n_ramp_down = n_steps - n_ramp_up
    # learnrates = np.hstack([
    #     np.linspace(0.1, 1.0, n_ramp_up),
    #     np.linspace(1.0, 0.1, n_ramp_down)
    # ])
    # momenta = np.hstack([
    #     np.linspace(0.999, 0.01, n_ramp_up),
    #     np.linspace(0.01, 0.999, n_ramp_down),
    # ])
    # print('n_steps', n_steps)
    # print('len(learnrates)', len(learnrates))

    # optimizer = OneCyclePolicy(
    #     net.parameters(),
    #     learnrates=learnrates,
    #     momenta=momenta,
    #     nesterov=True
    # )

    # this leads to the results from 2016 in ~5 epochs
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.25,
        momentum=0.9,
        weight_decay=1e-6, nesterov=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True
    )

    if cuda:
        net.cuda()

    train(
        cuda=cuda,
        run_path=run_path,
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=500,
        train_loader=train_loader,
        valid_loader=valid_loaders,
        logger=logger
    )


if __name__ == '__main__':
    main()
