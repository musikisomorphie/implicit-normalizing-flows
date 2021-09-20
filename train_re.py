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
import torch.nn.functional as F
import train_utils as utils
import lib.optimizers as optim

from lib.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

BERN_id = {1: (0, 0, 255),
           2: (255, 0, 0),
           3: (0, 128, 0)}

# Arguments
parser = argparse.ArgumentParser()
# deepspeed parameters
parser.add_argument('--backend', type=str, default='nccl',
                    help='distributed backend')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

# essential parameters for model learning
parser.add_argument('--dataset',
                    type=str,
                    default='scrc',
                    choices=['scrc', 'rxrx1'])
parser.add_argument('--dataroot', type=str, default='data')
parser.add_argument('--save', help='directory to save results',
                    type=str, default='experiment1')

parser.add_argument('--classifier', type=str, default='resnet',
                    choices=['resnet', 'densenet'])
parser.add_argument('--flow', type=str, default='reflow',
                    choices=['reflow', 'imflow'])
parser.add_argument('--shuffle-factor',
                    type=int,
                    default=2,
                    help='the factor signature for pixel_shuffle or pixel unshuffle')
parser.add_argument('--scale-factor',
                    type=float,
                    default=0.5,
                    help='the scale-factor signature for F.interpolate')
parser.add_argument('--env', type=str,
                    choices=['012', '120', '201'])
parser.add_argument('--inp', type=str,
                    choices=['i', 'mi'],
                    default='i')
parser.add_argument('--right-pad', type=int, default=0)
parser.add_argument('--imagesize', type=int, default=256)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=32)

parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--couple-label', action='store_true', default=False,
                    help='couple label in the transformation')
parser.add_argument('--factor-out', action='store_true', default=False,
                    help='half the output dimension for each block')
parser.add_argument('--actnorm', action='store_true', default=False,
                    help='add actnorm layer')

# useful parameters
parser.add_argument('--task', type=str,
                    choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--eval-batchsize',
                    help='minibatch size', type=int, default=200)
parser.add_argument(
    '--print-freq', help='Print progress every so iterations', type=int, default=120)
parser.add_argument(
    '--vis-freq', help='Visualize progress every so iterations', type=int, default=500)

# less essential parameters
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
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval,
                    choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--scale-dim', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str,
                    choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str,
                    choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

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


def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32).to(start) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]

    return out


def rev_proc_img(x,
                 y,
                 shuffle_factor,
                 couple_label):
    if couple_label:
        x, y_pred = x[:, :-1], x[:, -1]
        print(y_pred.shape)
        print('couple label diff', (y_pred -
                                    y.view(y.shape[0], 1, 1)).abs().max())
    x = torch.pixel_shuffle(x, shuffle_factor)
    return x


def compute_loss(x,
                 y,
                 lab,
                 model_clss,
                 model_symm,
                 criterion,
                 msk_len_z=0,
                 beta=1.0,
                 nvals=256,
                 is_train=True):
    x.requires_grad = True
    logits_tensor = model_clss(x)
    crossent = criterion(logits_tensor, lab)
    if is_train:
        grad_clss = torch.autograd.grad(outputs=crossent,
                                        inputs=x,
                                        retain_graph=is_train)[0]
    else:
        grad_clss = torch.ones_like(x)
    grad_clss = F.interpolate(grad_clss, scale_factor=0.25)
    grad_clss = torch.pixel_unshuffle(grad_clss, 2)

    x = x[:8]
    y = y[:8]
    grad_clss = grad_clss[:8]
    z, delta_logp = model_symm(x, y, [0, grad_clss])
    delta_logp, tot_grad = delta_logp
    grad_mean = tot_grad.norm(p=1, dim=1).mean()
    if msk_len_z:
        z = z[:, msk_len_z:]

    # log p(z)
    logpz = utils.standard_normal_logprob(z).sum(1, keepdim=True)

    # log p(x)
    logpx = logpz - beta * delta_logp - \
        np.log(nvals) * np.prod(z.shape[1:])

    bits_per_dim = - torch.mean(logpx) / \
        np.prod(z.shape[1:]) / np.log(2)

    logpz = torch.mean(logpz).detach()

    delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp, crossent, grad_mean


def train(args,
          device,
          epoch,
          model_clss,
          model_symm,
          optim_clss,
          optim_symm,
          criterion,
          trn_loader,
          logger,
          ema,
          scheduler,
          msk_len_z=0,
          cls_num_y=1):

    model_clss.train()
    model_symm.train()

    total = 0
    correct = 0

    end = time.time()

    for i, (x, y, meta) in enumerate(trn_loader):
        global_itr = epoch * len(trn_loader) + i
        utils.update_lr(optim_symm, global_itr,
                        args.warmup_iters, args.lr)

        model_clss.zero_grad()
        model_symm.zero_grad()

        x = x.to(device)
        y = y.to(device)

        beta = min(1, global_itr /
                   args.annealing_iters) if args.annealing_iters > 0 else 1.
        bpd, logits, logpz, neg_delta_logp, crossent, grad_mean = compute_loss(x, y / cls_num_y, y,
                                                                               model_clss, model_symm,
                                                                               criterion,
                                                                               msk_len_z, beta)
        # print(tot_grad.shape, tot_grad.abs().mean())
        # crossent = criterion(logits, y)

        # update a variety of metrics
        bpd_meter.update(bpd.item())
        ce_meter.update(crossent.item())
        logpz_meter.update(logpz.item())
        deltalogp_meter.update(neg_delta_logp.item())
        firmom, secmom = utils.estimator_moments(model_symm)
        firmom_meter.update(firmom)
        secmom_meter.update(secmom)

        # compute accuracy.
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        model_clss.backward(crossent)
        optim_clss.step()
        scheduler.step()

        model_symm.backward(bpd)
        if global_itr % args.update_freq == args.update_freq - 1:
            if args.update_freq > 1:
                with torch.no_grad():
                    for p in model_symm.parameters():
                        if p.grad is not None:
                            p.grad /= args.update_freq

            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model_symm.parameters(), 1.)
            gnorm_meter.update(grad_norm)

            if args.learn_p:
                utils.compute_p_grads(model_symm)

            optim_symm.step()
            utils.update_lipschitz(model_symm)
            ema.apply()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('mean of the gradient {}'.format(grad_mean))
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(trn_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
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
            save_kwargs = {'img_path': args.save / 'imgs',
                           'epoch': epoch,
                           'iteration': i,
                           'phase': 'TRN'}
            visualize(model_clss,
                      model_symm,
                      x,
                      y / cls_num_y,
                      args.scale_factor,
                      args.shuffle_factor,
                      args.couple_label,
                      msk_len_z,
                      cls_num_y,
                      **save_kwargs)

        del x
        torch.cuda.empty_cache()
        gc.collect()


def evaluate(args,
             device,
             epoch,
             model_clss,
             model_symm,
             criterion,
             eval_loader,
             logger,
             ema,
             phase,
             msk_len_z=0,
             cls_num_y=1,
             is_visualize=True):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    # if ema is not None:
    #     ema.swap()

    utils.update_lipschitz(model_symm)

    model_clss.eval()
    model_symm.eval()

    correct = 0
    total = 0

    start = time.time()
    for i, (x, y, meta) in enumerate(tqdm(eval_loader)):
        x = x.to(device)
        y = y.to(device)
        bpd, logits, _, _, loss, _ = compute_loss(x, y / cls_num_y, y,
                                                  model_clss, model_symm,
                                                  criterion,
                                                  msk_len_z,
                                                  is_train=False)
        bpd_meter.update(bpd.item(), x.size(0))

        if args.task in ['classification', 'hybrid']:
            # loss = criterion(logits, y)
            ce_meter.update(loss.item(), x.size(0))
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        if i % args.vis_freq == 0 and is_visualize:
            save_kwargs = {'img_path': args.save / 'imgs',
                           'epoch': epoch,
                           'iteration': i,
                           'phase': phase}
            visualize(model_clss,
                      model_symm,
                      x,
                      y / cls_num_y,
                      args.scale_factor,
                      args.shuffle_factor,
                      args.couple_label,
                      msk_len_z,
                      cls_num_y,
                      **save_kwargs)

    val_time = time.time() - start

    # if ema is not None:
    #     ema.swap()
    s = '{} | Epoch: [{}]\tTime {:.2f} | bits/dim {bpd_meter.avg:.4f}'.format(
        phase, epoch, val_time, bpd_meter=bpd_meter)
    if args.task in ['classification', 'hybrid']:
        s += ' | CE {:.4f} | Acc {:.4%}'.format(
            ce_meter.avg, correct / total)
    logger.info(s)
    return correct / total


def visualize(model_clss,
              model_symm,
              real_imgs,
              real_labs,
              scale_factor,
              shuffle_factor,
              couple_label,
              msk_len_z=0,
              cls_num_y=1,
              **save_kwargs):
    model_symm.eval()
    real_imgs = real_imgs[:8]
    real_labs = real_labs[:8]
    with torch.no_grad():
        # reconstructed real images
        real_z = model_symm(real_imgs, real_labs)
        if scale_factor != 1:
            real_imgs = F.interpolate(real_imgs,
                                      scale_factor=scale_factor)

        recn_imgs = model_symm(real_z, inverse=True)
        recn_imgs = rev_proc_img(recn_imgs,
                                 real_labs,
                                 shuffle_factor,
                                 couple_label)

        intp_z = linspace(real_z[0], real_z[-1], real_z.shape[0])
        intp_imgs = model_symm(intp_z, inverse=True)
        intp_imgs = rev_proc_img(intp_imgs,
                                 real_labs,
                                 shuffle_factor,
                                 couple_label)

        # random samples
        if msk_len_z:
            cell_ann = real_z[:, :msk_len_z]
            fake_imgs = model_symm(cell_ann, inverse=True)
            fake_imgs = rev_proc_img(fake_imgs,
                                     real_labs,
                                     shuffle_factor,
                                     couple_label)

            cell_ann = cell_ann.view(real_imgs.shape[0],
                                     -1,
                                     real_imgs.shape[2] // (shuffle_factor * 2),
                                     real_imgs.shape[3] // (shuffle_factor * 2))
            cell_ann = torch.pixel_shuffle(cell_ann, 2)
            cell_ann = torch.pixel_shuffle(cell_ann, shuffle_factor)

            diff_recn = (real_imgs[:, 0] - recn_imgs[:, 0]).abs().max()
            diff_fake = (real_imgs[:, 0] - fake_imgs[:, 0]).abs().max()
            diff_cell = (real_imgs[:, 0] - cell_ann.squeeze()).abs().max()
            print('diff {:10f}, {:10f}, {:10f}'.format(diff_recn,
                                                       diff_fake,
                                                       diff_cell))
        else:
            fake_z = torch.ones([real_imgs.shape[0]]).to(real_imgs)
            fake_imgs = model_symm(fake_z, inverse=True)
            fake_imgs = rev_proc_img(fake_imgs,
                                     real_labs,
                                     shuffle_factor,
                                     couple_label)

        imgs = torch.cat([real_imgs, recn_imgs, fake_imgs, intp_imgs], 0)
        imgs = imgs[:, -3:]

        if msk_len_z:
            real_cll_ann = cell_ann.clone().squeeze()
            real_cll_ann = torch.round(real_cll_ann * cls_num_y)
            fake_img_ann = fake_imgs[:, 1:].clone().permute(0, 2, 3, 1)
            for i in range(1, cls_num_y):
                fake_img_ann[real_cll_ann == i] = torch.tensor(
                    BERN_id[i]).to(fake_img_ann)

            imgs = torch.cat([imgs,
                              fake_img_ann.permute(0, 3, 1, 2)], 0)

            real_cll_ann = real_imgs[:, 0].clone().squeeze()
            real_cll_ann = torch.round(real_cll_ann * cls_num_y)
            fake_img_ann = fake_imgs[:, 1:].clone().permute(0, 2, 3, 1)
            for i in range(1, cls_num_y):
                fake_img_ann[real_cll_ann == i] = torch.tensor(
                    BERN_id[i]).to(fake_img_ann)

            imgs = torch.cat([imgs,
                              fake_img_ann.permute(0, 3, 1, 2)], 0)

        filename = pathlib.Path(save_kwargs['img_path']) / \
            'e{:03d}_i{:06d}_{}.png'.format(save_kwargs['epoch'],
                                            save_kwargs['iteration'],
                                            save_kwargs['phase'])
        tv_utils.save_image(imgs.cpu().float(),
                            str(filename),
                            nrow=8,
                            padding=2)
    model_symm.train()


def main(args):
    exp_config = ('{}_{}_{}_{}_{}_{}_'
                  '{}_{}_{}_{}_{}_{}_'
                  '{}_{}_{}').format(args.dataset,
                                     args.classifier,
                                     args.flow,
                                     args.shuffle_factor,
                                     args.scale_factor,
                                     args.env,
                                     args.inp,
                                     args.right_pad,
                                     args.imagesize,
                                     args.batchsize,
                                     args.nepochs,
                                     args.nblocks,
                                     args.couple_label,
                                     args.factor_out,
                                     args.actnorm)
    args.save = pathlib.Path(args.save) / exp_config
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

    logger.info('Loading dataset {}'.format(args.dataset))
    data_loader, cls_num_y, input_size = utils.data_prep(args)
    if args.dataset == 'rxrx1':
        trn_loader, val_loader, tst_loader, itst_loader = data_loader
    elif args.dataset == 'scrc':
        trn_loader, val_loader, tst_loader = data_loader
    logger.info('Dataset loaded with input size {}.'.format(input_size))

    logger.info('Creating model.')
    classifier = utils.PredNet(args.classifier, cls_num_y, input_size[1])
    param_clss = filter(lambda p: p.requires_grad, classifier.parameters())
    optim_clss = optim.Adam(classifier.parameters(),
                            lr=args.lr, weight_decay=1e-5)
    model_clss, optim_clss, _, __ = deepspeed.initialize(args=args,
                                                         model=classifier,
                                                         model_parameters=param_clss,
                                                         optimizer=optim_clss)

    symmetrier = utils.normflow(args, input_size)
    param_symm = filter(lambda p: p.requires_grad, symmetrier.parameters())
    optim_symm = optim.Adam(symmetrier.parameters(), lr=args.lr)
    model_symm, optim_symm, _, __ = deepspeed.initialize(args=args,
                                                         model=symmetrier,
                                                         model_parameters=param_symm,
                                                         optimizer=optim_symm)
    criterion = torch.nn.CrossEntropyLoss()

    # scheduler = None
    # if args.scheduler:
    #     scheduler = CosineAnnealingWarmRestarts(
    #         optimizer, 20, T_mult=2, last_epoch=args.begin_epoch - 1)
    scheduler = get_cosine_schedule_with_warmup(
        optim_clss,
        num_training_steps=len(trn_loader) * args.nepochs,
        num_warmup_steps=5415)

    ords = []
    last_checkpoints = []
    lipschitz_constants = []
    best_val_bpd = -1e4
    ema = utils.ExponentialMovingAverage(model_symm)
    msk_len_z = 0
    if args.dataset == 'scrc' and args.inp == 'mi':
        msk_len_z = int((args.imagesize * args.scale_factor) ** 2)
    for epoch in range(args.begin_epoch, args.nepochs):
        train(args,
              device,
              epoch,
              model_clss,
              model_symm,
              optim_clss,
              optim_symm,
              criterion,
              trn_loader,
              logger,
              ema,
              scheduler,
              msk_len_z,
              cls_num_y)

        lipschitz_constants.append(utils.get_lipschitz_constants(model_symm))
        ords.append(utils.get_ords(model_symm))
        # logger.info('Lipsh: {}'.format(
        #     utils.pretty_repr(lipschitz_constants[-1])))
        # logger.info('Order: {}'.format(utils.pretty_repr(ords[-1])))

        val_bpd = evaluate(args,
                           device,
                           epoch,
                           model_clss,
                           model_symm,
                           criterion,
                           val_loader,
                           logger,
                           ema if args.ema_val else None,
                           'VAL',
                           msk_len_z,
                           cls_num_y)

        if val_bpd > best_val_bpd:
            msg = 'Save the checkout points for the best evaluation results \n'
            msg += 'VAL: {:.4%} '.format(val_bpd)
            best_val_bpd = val_bpd
            torch.save(model_clss.state_dict(),
                       args.save / 'classifier_{}.pth'.format(epoch))
            torch.save(model_symm.state_dict(),
                       args.save / 'symmetrier_{}.pth'.format(epoch))

            tst_bpd = evaluate(args,
                               device,
                               epoch,
                               model_clss,
                               model_symm,
                               criterion,
                               tst_loader,
                               logger,
                               ema if args.ema_val else None,
                               'TST',
                               msk_len_z,
                               cls_num_y,
                               is_visualize=False)
            msg += 'TEST: {:.4%} '.format(tst_bpd)

            if args.dataset == 'rxrx1':
                itst_bpd = evaluate(args,
                                    device,
                                    epoch,
                                    model_clss,
                                    model_symm,
                                    criterion,
                                    itst_loader,
                                    logger,
                                    ema if args.ema_val else None,
                                    'ITST',
                                    msk_len_z,
                                    cls_num_y,
                                    is_visualize=False)
                msg += 'ID_TEST: {:.4%} '.format(itst_bpd)
            logger.info(msg)

    torch.cuda.synchronize()


if __name__ == '__main__':
    main(args)
