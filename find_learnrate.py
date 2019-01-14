from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader

import datasets
import models

from utils import filenames_from_splitfile
from utils import ChunkedRandomSampler
from utils import find_learnrate
from utils import OneCyclePolicy
import os

import numpy as np

import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='train a VGG style conv net on the MAPS dataset')
    parser.add_argument('splits', type=str, help='on which splits to train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=1024)
    parser.add_argument('--plotname', type=str,
                        default='find_learnrate.pdf',
                        help='the name of the plot')
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

    ##########################################
    # prepare train data
    train_filenames = filenames_from_splitfile(os.path.join(args.splits, 'train'))
    train_sequences = datasets.get_sequences(train_filenames)

    batch_size = args.batch_size
    n_steps = args.n_steps
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

    net = models.VGGNet2016()
    optimizer = OneCyclePolicy(
        net.parameters(),
        learnrates=np.linspace(
            0.0001,
            10,
            n_steps
        ),
        momenta=np.ones(n_steps) * 0.9,
        nesterov=True
    )

    if cuda:
        net.cuda()

    losses, learning_rates = find_learnrate(
        cuda=cuda,
        net=net,
        optimizer=optimizer,
        loader=train_loader
    )

    fig, ax = plt.subplots()
    ax.semilogx(learning_rates, losses)
    ax.set_xlabel('learning rate (log scale)')
    ax.set_ylabel('loss')
    fig.tight_layout()
    fig.savefig(args.plotname)


if __name__ == '__main__':
    main()
