import os
import shutil
import torch
from torch import nn
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from torch.utils.data.sampler import Sampler
from torch.optim import SGD


def ensure_empty_directory_exists(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def filenames_from_splitfile(split_file):
    filenames = open(split_file, 'r').readlines()
    return [f.strip() for f in filenames]


def train(cuda, run_path, net, optimizer, scheduler, n_epochs, train_loader, valid_loader, logger):
    epoch = 0

    best_valid_f = -np.inf
    best_valid_loss = np.inf

    current_net_filename = os.path.join(run_path, 'current_net_state.pkl')
    current_optimizer_filename = os.path.join(run_path, 'current_optimizer_state.pkl')
    current_scheduler_filename = os.path.join(run_path, 'current_scheduler_state.pkl')

    best_valid_loss_net_filename = os.path.join(run_path, 'best_valid_loss_net_state.pkl')
    best_valid_f_net_filename = os.path.join(run_path, 'best_valid_f_net_state.pkl')

    for epoch in range(n_epochs):
        #############
        print('epoch {}/{}'.format(epoch, n_epochs))
        print('training...')
        train_loss = train_one_epoch(cuda, net, optimizer, train_loader)
        logger.add_scalar('train/loss', train_loss, global_step=epoch)

        #############
        print('validating...')
        valid_loss, p, r, f = evaluate(cuda, net, valid_loader)
        print('l {:8.4f} p {:4.2f}, r {:4.2f}, f {:4.2f}'.format(valid_loss, p, r, f))
        logger.add_scalar('valid/loss', valid_loss, global_step=epoch)
        logger.add_scalar('valid/p', p, global_step=epoch)
        logger.add_scalar('valid/r', r, global_step=epoch)
        logger.add_scalar('valid/f', f, global_step=epoch)

        #############
        # always save current state
        torch.save(net.state_dict(), current_net_filename)
        torch.save(optimizer.state_dict(), current_optimizer_filename)
        torch.save(scheduler.state_dict(), current_scheduler_filename)

        # save net state when we get better validation loss
        if valid_loss < best_valid_loss:
            torch.save(net.state_dict(), best_valid_loss_net_filename)

        # save net state when we get better validation f-measure
        if f > best_valid_f:
            torch.save(net.state_dict(), best_valid_f_net_filename)

        # always recorded, if present, to keep track of lr-scheduler
        for gi, param_group in enumerate(optimizer.param_groups):
            if 'lr' in param_group:
                logger.add_scalar(
                    'train/lr',
                    param_group['lr'],
                    global_step=epoch
                )
            if 'momentum' in param_group:
                logger.add_scalar(
                    'train/momentum',
                    param_group['momentum'],
                    global_step=epoch
                )

        # step on the validation loss only
        scheduler.step(valid_loss)
        epoch += 1


def train_one_epoch(cuda, net, optimizer, loader):
    net.train()
    loss_function = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
    smoothed_loss = 1.
    t_elapsed = 0
    n_batches = float(len(loader))
    current_count = 0
    total_count = 0
    for x, y in loader:
        t_start = time.time()
        if cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        y_hat = net.forward(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()
        current_count += 1
        total_count += 1
        smoothed_loss = smoothed_loss * 0.9 + loss.detach().cpu().item() * 0.1

        # bail if NaN or Inf is encountered
        if np.isnan(smoothed_loss) or np.isinf(smoothed_loss):
            print('encountered NaN/Inf in smoothed_loss "{}"'.format(smoothed_loss))
            exit(-1)

        t_end = time.time()
        t_elapsed += (t_end - t_start)
        if t_elapsed > 60:
            batches_per_second = current_count / t_elapsed
            t_rest = ((n_batches - total_count) / batches_per_second) / 3600.
            print('bps {:4.2f} eta {:4.2f} [h]'.format(batches_per_second, t_rest))
            t_elapsed = 0
            current_count = 0
    return smoothed_loss


def find_learnrate(cuda, net, optimizer, loader):
    net.train()
    loss_function = nn.BCEWithLogitsLoss(reduction='elementwise_mean')

    t_elapsed = 0
    n_batches = float(len(loader))
    current_count = 0
    total_count = 0

    losses = []
    lrs = []
    for i_batch, (x, y) in enumerate(loader):
        t_start = time.time()
        if cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        y_hat = net.forward(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()
        current_count += 1
        total_count += 1
        loss = loss.detach().cpu().item()

        losses.append(loss)
        lrs.append(optimizer.current_lr())

        # bail if NaN or Inf is encountered
        if np.isnan(loss) or np.isinf(loss):
            return losses, lrs

        t_end = time.time()
        t_elapsed += (t_end - t_start)
        if t_elapsed > 60:
            batches_per_second = current_count / t_elapsed
            t_rest = ((n_batches - total_count) / batches_per_second) / 3600.
            print('bps {:4.2f} eta {:4.2f} [h]'.format(batches_per_second, t_rest))
            t_elapsed = 0
            current_count = 0
    return losses, lrs


def evaluate(cuda, net, loader):
    if isinstance(loader, list):
        return evaluate_multiple_loaders(cuda, net, loader)
    else:
        return evaluate_one_loader(cuda, net, loader)


def evaluate_multiple_loaders(cuda, net, loaders):
    valid_loss, p, r, f = 0, 0, 0, 0

    for loader in loaders:
        i_vl, i_p, i_r, i_f = evaluate_one_loader(cuda, net, loader)
        valid_loss += i_vl
        p += i_p
        r += i_r
        f += i_f

    n = float(len(loaders))
    valid_loss /= n
    p /= n
    r /= n
    f /= n
    return valid_loss, p, r, f


def evaluate_one_loader(cuda, net, loader):
    net.eval()
    loss_function = nn.BCELoss(reduction='elementwise_mean')
    smoothed_loss = 1.
    y_true = []
    y_pred = []
    for x, y in loader:
        if cuda:
            x = x.cuda()
            y = y.cuda()

        y_hat = net.predict(x)
        loss = loss_function(y_hat, y)
        smoothed_loss = smoothed_loss * 0.9 + loss.detach().cpu().item() * 0.1

        y_true.append(y.detach().cpu().numpy())
        y_pred.append((y_hat.detach().cpu().numpy() > 0.5) * 1)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    p, r, f, _ = prfs(y_true, y_pred, average='micro')
    return smoothed_loss, p, r, f


class ChunkedRandomSampler(Sampler):
    """Splits a dataset into smaller chunks (mainly to re-define what is considered an 'epoch').
       Samples elements randomly from a given list of indices, without replacement.
       If a chunk would be underpopulated, it's filled up with rest-samples.

    Arguments:
        data_source (Dataset): a dataset
        chunk_size      (int): how large a chunk should be
    """

    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.i = 0
        self.N = len(self.data_source)
        # re-did this as numpy permutation, b/c FramedSignals do not like
        # torch tensors as indices ...
        # self.perm = torch.randperm(self.N)
        self.perm = np.random.permutation(self.N)

    def __iter__(self):
        rest = len(self.perm) - (self.i + self.chunk_size)
        if rest == 0:
            self.i = 0
            self.perm = np.random.permutation(self.N)
        elif rest < 0:
            # works b/c rest is negative
            carryover = self.chunk_size + rest
            self.i = 0
            self.perm = np.hstack([self.perm[-carryover:], np.random.permutation(self.N)])

        chunk = self.perm[self.i: self.i + self.chunk_size]
        self.i += self.chunk_size
        return iter(chunk)

    def __len__(self):
        return self.chunk_size


class OneCyclePolicy(SGD):
    def __init__(self, params, learnrates, momenta, dampening=0,
                 weight_decay=0, nesterov=False):
        super().__init__(
            params,
            learnrates.min(),
            momenta.min(),
            dampening,
            weight_decay,
            nesterov
        )
        self.__i_step = -1
        self.__learnrates = learnrates
        self.__momenta = momenta

    def current_lr(self):
        return self.__learnrates[self.__i_step]

    def current_momentum(self):
        return self.__momenta[self.__i_step]

    def next_learnrate(self):
        self.__i_step = (self.__i_step + 1) % len(self.__learnrates)

    def step(self, closure=None):
        self.next_learnrate()
        self.param_groups[0]['lr'] = self.current_lr()
        self.param_groups[0]['momentum'] = self.current_momentum()
        return super().step(closure)
