from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

import datasets
import models
import torch

from utils import evaluate_multiple_loaders
from utils import filenames_from_splitfile

import argparse


def main():
    parser = argparse.ArgumentParser(description='evaluate a convolutional network on the MAPS dataset')
    parser.add_argument('model', choices=models.get_model_classes(),
                        help='any classname of model as defined in "models.py"')
    parser.add_argument('net_state', type=str,
                        help='which network state to use')
    parser.add_argument('splitfile', type=str,
                        help='the list of files that will be evaluated')
    parser.add_argument('--start_end', type=str, default="0,750", help='start and end frame; the default value boils down to the first 30[s] at a framerate of 25[fps]')
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

    # prepare test data as individual sequences
    test_filenames = filenames_from_splitfile(args.splitfile)

    start_end = None
    try:
        if args.start_end is not None:
            start, end = tuple(int(t) for t in args.start_end.split(','))
            duration = end - start
            if duration <= 0:
                raise ValueError('negative durations are unacceptable ...')
            duration_in_s = float(duration) / 25.
        print('evaluating on frames {} of the data ({} [s])'.format((start, end), duration_in_s))
        start_end = (start, end)
        if abs(duration_in_s - 30.) < 0.5:
            print('using the SAME evaluation protocol (as in the paper)')
        else:
            print('using a DIFFERENT evalution protocol (not as in the paper)')
    except Exception as e:
        print('evaluating on ALL the data')
        print('using a DIFFERENT evalution protocol (not as in the paper)')

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

    net_class = getattr(models, args.model, None)
    if net_class is None:
        raise RuntimeError('could not find model class named "{}" in "models.py"'.format(args.model))

    net = net_class()
    net.load_state_dict(torch.load(args.net_state))

    if cuda:
        net.cuda()

    loss, p, r, f = evaluate_multiple_loaders(cuda, net, test_loaders)
    print('l {:8.4f} p {:8.4f}, r {:8.4f}, f {:8.4f}'.format(loss, p, r, f))


if __name__ == '__main__':
    main()
