from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

import datasets
import models
import torch

from utils import evaluate_multiple_loaders
from utils import filenames_from_splitfile

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='train a VGG style conv net on the MAPS dataset')
    parser.add_argument('net_state', type=str,
                        help='which network state to use')
    parser.add_argument('splits', type=str,
                        help='splits directory from which the "test" split will be taken')
    parser.add_argument('--start_end', type=str, default="0,750")
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

    # prepare valid data as individual sequences
    test_filenames = filenames_from_splitfile(os.path.join(args.splits, 'test'))

    # to get all of the sequence data, just write this:
    start_end = None
    try:
        if args.start_end is not None:
            start_end = tuple(int(t) for t in args.start_end.split(','))
    except Exception as e:
        print('ERROR: invalid input for "start_end"!')
        pass
    test_sequences = datasets.get_sequences(test_filenames, start_end)

    test_loaders = []
    for test_sequence in test_sequences:
        test_loader = DataLoader(
            test_sequence,
            batch_size=1024,
            sampler=SequentialSampler(test_sequence),
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        test_loaders.append(test_loader)

    net = models.VGGNet2016()
    net.load_state_dict(torch.load(args.net_state))

    if cuda:
        net.cuda()

    loss, p, r, f = evaluate_multiple_loaders(cuda, net, test_loaders)
    print('l {:8.4f} p {:8.4f}, r {:8.4f}, f {:8.4f}'.format(loss, p, r, f))


if __name__ == '__main__':
    main()
