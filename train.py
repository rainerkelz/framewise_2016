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
# from utils import OneCyclePolicy
import os

import argparse


def main():
    parser = argparse.ArgumentParser(description=
                                     'train a convolutional network' +
                                     'on the MAPS dataset')
    parser.add_argument('splits', type=str,
                        help='on which splits to train')
    parser.add_argument('run_path', type=str,
                        help='where to write run state')
    parser.add_argument('model', choices=models.get_model_classes(),
                        help='any classname of model as defined in "models.py"')
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

    train_dataset = ConcatDataset(train_sequences)

    batch_size = 128
    n_epochs = 500

    # go with the original definition of an 'epoch' (aka the whole dataset ...)
    # as opposed to 'arbitrary number of steps until we validate'
    # n_steps = 8192
    n_steps = len(train_dataset) // batch_size
    print('adjusted n_steps', n_steps)

    # one train loader for all train sequences
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

    net_class = getattr(models, args.model, None)
    if net_class is None:
        raise RuntimeError('could not find model class named "{}" in "models.py"'.format(
            args.model
        ))

    net = net_class()
    if args.model == 'AllConv2016':
        print('choosing AllConv2016 learnrate and learnrate schedule!')
        # this does not train all that well ... validation loss stays high all the time?
        optimizer = optim.SGD(
            net.parameters(),
            lr=1.0,
            momentum=0.9,
            weight_decay=1e-5,
            nesterov=False
        )
        milestones = list(range(10, n_epochs, 10))
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            gamma=0.5
        )
    else:
        print('choosing VGGNet2016 learnrate and learnrate schedule!')
        # this leads to the results from 2016 in ~5 epochs
        optimizer = optim.SGD(
            net.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-5,
            nesterov=False
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True,
            min_lr=1e-4
        )

    if cuda:
        net.cuda()

    train(
        cuda=cuda,
        run_path=run_path,
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        train_loader=train_loader,
        valid_loader=valid_loaders,
        logger=logger
    )


if __name__ == '__main__':
    main()
