import deepspeed
import argparse
import time
import math
import os
import os.path
import gc
import pathlib
import torch

import numpy as np
import torchvision.utils as tv_utils
import train_utils as utils
import lib.optimizers as optim

from lib.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default='nccl',
                    help='distributed backend')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--env', type=str,
                    choices=['012', '120', '201'])
parser.add_argument('--aug', type=str,
                    choices=['r', 'rr'])
parser.add_argument('--inp', type=str,
                    choices=['i', 'im', 'm'])
parser.add_argument('--oup', type=str,
                    choices=['cms'], default='cms')

parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'mnist',
        'cifar10',
        'svhn',
        'celebahq',
        'celeba_5bit',
        'imagenet32',
        'imagenet64',
        'scrc'
    ]
)
parser.add_argument('--classifier', type=str, default='resnet',
                    choices=['resnet', 'densenet'])
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--dataroot', type=str, default='data')
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

parser.add_argument('--block', type=str,
                    choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval,
                    choices=[True, False], default=False)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval,
                    choices=[True, False], default=False)
parser.add_argument(
    '--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval,
                    choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval,
                    choices=[True, False], default=True)

parser.add_argument('--act', type=str, default='swish')
parser.add_argument('--idim', type=int, default=512)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval,
                    default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval,
                    choices=[True, False], default=True)
parser.add_argument('--quadratic', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)
parser.add_argument('--preact', type=eval,
                    choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval,
                    choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)

parser.add_argument('--optimizer', type=str,
                    choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval,
                    choices=[True, False], default=False)
parser.add_argument(
    '--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results',
                    type=str, default='experiment1')
parser.add_argument('--val-batchsize',
                    help='minibatch size', type=int, default=200)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval,
                    choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str,
                    choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--scale-dim', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str,
                    choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str,
                    choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument(
    '--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument(
    '--vis-freq', help='Visualize progress every so iterations', type=int, default=500)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

batch_time = utils.RunningAverageMeter(0.97)
bpd_meter = utils.RunningAverageMeter(0.97)
logpz_meter = utils.RunningAverageMeter(0.97)
deltalogp_meter = utils.RunningAverageMeter(0.97)
firmom_meter = utils.RunningAverageMeter(0.97)
secmom_meter = utils.RunningAverageMeter(0.97)
gnorm_meter = utils.RunningAverageMeter(0.97)
ce_meter = utils.RunningAverageMeter(0.97)


def compute_loss(x, y,
                 model,
                 beta=1.0,
                 nvals=256):

    bits_per_dim = torch.zeros(1).to(x)
    logits_tensor = None
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    if args.task == 'hybrid':
        z_logp, logits_tensor = model((x, y), 0, classify=True)
        z, delta_logp = z_logp
    elif args.task == 'density':
        z, delta_logp = model((x, y), 0)
    elif args.task == 'classification':
        z, logits_tensor = model((x, y), classify=True)

    if args.task in ['density', 'hybrid']:
        # log p(z)
        logpz = utils.standard_normal_logprob(z).sum(1, keepdim=True)

        # log p(x)
        logpx = logpz - beta * delta_logp - \
            np.log(nvals) * np.prod(z.shape[1:])

        bits_per_dim = - torch.mean(logpx) / \
            np.prod(z.shape[1:]) / np.log(2)

        logpz = torch.mean(logpz).detach()

        delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp


def train(args,
          device,
          epoch,
          model,
          criterion,
          trn_loader,
          optimizer,
          logger,
          ema):

    model.train()

    total = 0
    correct = 0

    end = time.time()

    trn_iter = iter(trn_loader[0])
    for i, (x_1, y_1) in enumerate(trn_loader[1]):
        global_itr = epoch * len(trn_loader[1]) + i
        utils.update_lr(optimizer, global_itr,
                        args.warmup_iters, args.lr)

        optimizer.zero_grad()
        try:
            (x_0, y_0) = next(trn_iter)
        except StopIteration:
            trn_iter = iter(trn_loader[0])
            (x_0, y_0) = next(trn_iter)

        x = torch.cat((x_0, x_1), dim=0)
        y = torch.cat((y_0, y_1), dim=0)
        bat_id = np.random.rand(x.shape[0]).argsort()
        x = x[bat_id, ]
        y = y[bat_id, ]
        x = x.to(device)
        y = y.to(device)

        beta = min(1, global_itr /
                   args.annealing_iters) if args.annealing_iters > 0 else 1.
        bpd, logits, logpz, neg_delta_logp = compute_loss(
            x, y, model, beta=beta)

        if args.task in ['density', 'hybrid']:
            firmom, secmom = utils.estimator_moments(model)

            bpd_meter.update(bpd.item())
            logpz_meter.update(logpz.item())
            deltalogp_meter.update(neg_delta_logp.item())
            firmom_meter.update(firmom)
            secmom_meter.update(secmom)

        if args.task in ['classification', 'hybrid']:
            crossent = criterion(logits, y)
            ce_meter.update(crossent.item())

            # Compute accuracy.
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        # compute gradient and do SGD step
        if args.task == 'density':
            loss = bpd
        elif args.task == 'classification':
            loss = crossent
        else:
            # Change cross entropy from nats to bits.
            loss = bpd + crossent / np.log(2)
        loss.backward()

        if global_itr % args.update_freq == args.update_freq - 1:
            if args.update_freq > 1:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= args.update_freq

            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), 1.)
            if args.learn_p:
                utils.compute_p_grads(model)

            optimizer.step()
            utils.update_lipschitz(model)
            ema.apply()

            gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(trn_loader[1]), batch_time=batch_time, gnorm_meter=gnorm_meter
                )
            )

            if args.task in ['density', 'hybrid']:
                s += (
                    ' | Bits/dim {bpd_meter.val:.4f}({bpd_meter.avg:.4f}) | '
                    'Logpz {logpz_meter.avg:.0f} | '
                    '-DeltaLogp {deltalogp_meter.avg:.0f} | '
                    'EstMoment ({firmom_meter.avg:.0f},{secmom_meter.avg:.0f})'.format(
                        bpd_meter=bpd_meter, logpz_meter=logpz_meter, deltalogp_meter=deltalogp_meter,
                        firmom_meter=firmom_meter, secmom_meter=secmom_meter
                    )
                )

            if args.task in ['classification', 'hybrid']:
                s += ' | CE {:.4f} | Acc {:.4%}'.format(
                    ce_meter.avg, correct / total)

            logger.info(s)
        if i % args.vis_freq == 0:
            visualize(epoch, model, i, x, y, args.save / 'imgs')

        del x
        torch.cuda.empty_cache()
        gc.collect()


def validate(args,
             device,
             epoch,
             model,
             criterion,
             dat_loader,
             phase,
             logger,
             ema=None):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    utils.update_lipschitz(model)

    model.eval()

    correct = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dat_loader)):
            x = x.to(device)
            y = y.to(device)
            bpd, logits, _, _ = compute_loss(x, y, model)
            bpd_meter.update(bpd.item(), x.size(0))

            if args.task in ['classification', 'hybrid']:
                loss = criterion(logits, y)
                ce_meter.update(loss.item(), x.size(0))
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
    val_time = time.time() - start

    if ema is not None:
        ema.swap()
    s = '{} | Epoch: [{}]\tTime {:.2f} | bits/dim {bpd_meter.avg:.4f}'.format(
        phase, epoch, val_time, bpd_meter=bpd_meter)
    if args.task in ['classification', 'hybrid']:
        s += ' | CE {:.4f} | Acc {:.4%}'.format(
            ce_meter.avg, correct / total)
    logger.info(s)
    return bpd_meter.avg


def visualize(epoch, model, itr, real_imgs, real_labs, img_path, nvals=256):
    model.eval()
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs

    with torch.no_grad():
        # reconstructed real images
        recon_imgs = model(model((real_imgs, real_labs)), inverse=True)

        # random samples
        fake_imgs = model(torch.Tensor([False]).to(real_imgs), inverse=True)

        imgs = torch.cat([_real_imgs, fake_imgs, recon_imgs], 0)
        print('visualize', imgs.shape, _real_imgs.shape,
              fake_imgs.shape, recon_imgs.shape)

        filename = pathlib.Path(img_path) / \
            'e{:03d}_i{:06d}.png'.format(epoch, itr)
        tv_utils.save_image(imgs.cpu().float(), str(
            filename), nrow=16, padding=2)
    model.train()


def main(args):
    args.save = pathlib.Path(args.save) / \
        '{}_{}_{}_{}_{}'.format(args.env, args.aug, args.inp,
                                args.imagesize, args.batchsize)
    (args.save / 'imgs').mkdir(parents=True, exist_ok=True)
    # logger
    logger = utils.get_logger(logpath=os.path.join(
        str(args.save), 'logs'), filepath=os.path.abspath(__file__))

    # Random seed
    if args.seed is None:
        args.seed = np.random.randint(100000)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.local_rank)
    torch.backends.cudnn.benchmark = True

    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info('{} \t Memory: {:.2f}GB'.format(
                props.name, props.total_memory / (1024**3)))
    else:
        logger.info('WARNING: Using device {}'.format(device))

    logger.info('Loading dataset {}'.format(args.data))
    trn_loader, tst_loader, n_classes, input_size = utils.data_prep(args)
    logger.info('Dataset loaded with input size {}.'.format(input_size))

    logger.info('Creating model.')
    model = utils.model_prep(args, 'reflow', input_size, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(args=args,
                                                   model=model,
                                                   model_parameters=parameters,
                                                   optimizer=optimizer)

    scheduler = None
    if args.scheduler:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 20, T_mult=2, last_epoch=args.begin_epoch - 1)
    criterion = torch.nn.CrossEntropyLoss()

    ords = []
    last_checkpoints = []
    lipschitz_constants = []
    best_val_bpd = math.inf
    ema = utils.ExponentialMovingAverage(model)
    for epoch in range(args.begin_epoch, args.nepochs):
        train(args,
              device,
              epoch,
              model,
              criterion,
              trn_loader,
              optimizer,
              logger,
              ema)

        lipschitz_constants.append(utils.get_lipschitz_constants(model))
        ords.append(utils.get_ords(model))
        logger.info('Lipsh: {}'.format(
            utils.pretty_repr(lipschitz_constants[-1])))
        # logger.info('Order: {}'.format(utils.pretty_repr(ords[-1])))

        val_bpd = validate(args,
                           device,
                           epoch,
                           model,
                           criterion,
                           tst_loader[0],
                           'VAL',
                           logger,
                           ema=ema if args.ema_val else None)

        if args.scheduler and scheduler is not None:
            scheduler.step()

        # if val_bpd < best_val_bpd:
        #     best_val_bpd = val_bpd
        #     utils.save_checkpoint({
        #         'state_dict': model.module.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'args': args,
        #         'ema': ema,
        #         'val_bpd': val_bpd,
        #     }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)

        # torch.save({
        #     'state_dict': model.module.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'args': args,
        #     'ema': ema,
        #     'val_bpd': val_bpd,
        # }, os.path.join(args.save, 'models', 'most_recent.pth'))

        # if args.ema_val:
        #     tst_bpd = validate(epoch, model, tst_loader[1], 'TST', ema)
        # else:
        #     tst_bpd = validate(epoch, model, tst_loader[1], 'TST')

    torch.cuda.synchronize()


if __name__ == '__main__':
    main(args)
