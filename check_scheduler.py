import torch
from torch import optim
import matplotlib.pyplot as plt


def get_multistep():
    x = torch.Tensor(1).uniform_() * 100.
    x.requires_grad = True

    optimizer = optim.SGD([x], lr=1.0)
    milestones = list(range(5, 100, 5))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    return x, optimizer, scheduler


def get_reduce():
    x = torch.Tensor(1).uniform_() * 100.
    x.requires_grad = True

    optimizer = optim.SGD([x], lr=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return x, optimizer, scheduler


def test(func):
    n_epochs = 100
    x, opt, sched = func()

    lrs = []
    losses = []
    valid_losses = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = x ** 2
        loss.backward()
        opt.step()
        if epoch < 10:
            fake_valid_loss = 1.
        elif epoch < 20:
            fake_valid_loss = 0.9
        elif epoch < 30:
            fake_valid_loss = 0.8
        else:
            fake_valid_loss = 0.5
        losses.append(loss)
        valid_losses.append(fake_valid_loss)

        lr = float('Inf')
        for gi, param_group in enumerate(opt.param_groups):
            if 'lr' in param_group:
                lr = param_group['lr']

        lrs.append(lr)

        # step on the validation loss only
        if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(fake_valid_loss, epoch=epoch)
        else:
            sched.step(epoch=epoch)

    return lrs, losses, valid_losses


def main():
    fig, axes = plt.subplots(nrows=3, sharex=True)

    lrs_reduce, losses_reduce, valid_losses_reduce = test(get_reduce)
    lrs_multi, losses_multi, valid_losses_multi = test(get_multistep)

    ax = axes[0]
    ax.set_title('losses')
    ax.plot(losses_reduce, label='losses_reduce')
    ax.plot(losses_multi, label='losses_multistep')

    ax = axes[1]
    ax.set_title('valid_losses')
    ax.plot(valid_losses_reduce, label='valid_losses_reduce')
    ax.plot(valid_losses_multi, label='valid_losses_multistep')
    ax.legend()

    ax = axes[2]
    ax.set_title('lrs')
    ax.plot(lrs_reduce, label='lrs_reduce')
    ax.plot(lrs_multi, label='lrs_multi')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
