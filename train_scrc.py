import argparse
import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc
import pathlib
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets

from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts
# import deepspeed


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

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
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
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
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

parser.add_argument('--nworkers', type=int, default=8)
parser.add_argument(
    '--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument(
    '--vis-freq', help='Visualize progress every so iterations', type=int, default=500)
# parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def compute_acc(logits, labs,
                met, n_class):

    _, prds = logits.max(1)
    for cms in range(n_class + 1):
        if cms < n_class:
            lcms = labs[labs == cms]
            pcms = prds[labs == cms]
            met[0, cms] += (prds == cms).sum().item()
        else:
            lcms = labs
            pcms = prds

        met[1, cms] += lcms.eq(pcms).sum().item()
        met[2, cms] += lcms.size(0)


def print_msg(logger, met, epoch, phase, n_clas, prefix='', eps=1e-5):
    msg = prefix + '[{}] Epoch: {} | Acc: {:.4%} '. \
        format(phase, epoch, met[1, -1] / (met[2, -1] + eps))
    for cms in range(n_clas):
        msg += '| CMS_{}: {:.4%}, {:.4%}, {:.2f} '. \
            format(cms + 1,
                   met[1, cms] / (met[2, cms] + eps),
                   met[0, cms] / (met[0, :-1].sum() + eps),
                   met[0, cms])
    logger.info(msg)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

if args.aug == 'r':
    trn_trans = transforms.Compose([
        transforms.RandomCrop(args.imagesize),
    ])
elif args.aug == 'rr':
    trn_trans = transforms.Compose([
        transforms.RandomCrop(args.imagesize),
        # utils.HEDJitter(0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
    ])

tst_trans = transforms.Compose([
    transforms.CenterCrop(args.imagesize)
])

if args.inp == 'i':
    scrc_in = [0, 1, 2]
elif args.inp == 'im':
    scrc_in = [0, 1, 2, 4]
elif args.inp == 'm':
    scrc_in = [4]

scrc_out = args.oup
if scrc_out == 'cms':
    n_classes = 4

trn_reg = args.env[:2]
tst_reg = args.env[-1]

dat_path = str(pathlib.Path(args.dataroot) / 'scrc_symm_{}.pt')
trn_data, trn_loader = list(), list()
for trn in trn_reg:
    trn_data.append(datasets.SCRC(dat_path.format(trn),
                                  scrc_in=scrc_in,
                                  scrc_out=scrc_out,
                                  transforms=trn_trans))
    trn_loader.append(torch.utils.data.DataLoader(trn_data[-1],
                                                  batch_size=args.batchsize,
                                                  shuffle=True,
                                                  num_workers=args.nworkers,
                                                  drop_last=True))

tst_size = 384
tst_path = dat_path.format(tst_reg)
tst_len = torch.load(str(tst_path))[0].shape[0]
print('test data size {}'.format(tst_len))
tst_idx = np.random.rand(tst_len).argsort()
tst_data, tst_loader = list(), list()
for i in range(1):
    # tst_sub_idx = tst_idx[:tst_size] if i == 0 else tst_idx[tst_size:]
    tst_sub_idx = tst_idx
    tst_data.append(datasets.SCRC(tst_path,
                                  tst_sub_idx,
                                  scrc_in,
                                  scrc_out,
                                  transforms=tst_trans))

    tst_loader.append(torch.utils.data.DataLoader(tst_data[-1],
                                                  batch_size=args.val_batchsize,
                                                  shuffle=False,
                                                  num_workers=args.nworkers,
                                                  drop_last=True))


input_size = (args.batchsize, len(scrc_in),
              args.imagesize, args.imagesize)
# dataset_size = len(train_loader.dataset)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 16,
                  input_size[2] // 4, input_size[3] // 4)
    squeeze_layer = layers.SqueezeLayer(4)

model = utils.initialize_model(args.classifier,
                               num_classes=n_classes,
                               chn_dim=len(scrc_in) * 4 if args.squeeze_first else len(scrc_in))

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()


save_path = pathlib.Path(args.save) / \
    '{}_{}_{}_{}_{}'.format(args.env, args.aug, args.inp,
                            args.imagesize, args.batchsize)
save_path.mkdir(parents=True, exist_ok=True)

logger = utils.custom_logger(str(save_path / 'train.log'))
best_trn_0 = np.zeros((3, n_classes + 1))
best_trn_1 = np.zeros_like(best_trn_0)
best_tst = np.zeros_like(best_trn_0)
best_epoch = 0.
for epoch in range(args.begin_epoch, args.nepochs):
    # print(epoch)
    model.train()
    met_0 = np.zeros_like(best_trn_0)
    met_1 = np.zeros_like(best_trn_0)
    trn_iter = iter(trn_loader[0])
    for i, (x_1, y_1) in enumerate(trn_loader[0]):
        try:
            (x_0, y_0) = next(trn_iter)
        except StopIteration:
            trn_iter = iter(trn_loader[1])
            (x_0, y_0) = next(trn_iter)

        optimizer.zero_grad()

        x = torch.cat((x_0, x_1), dim=0)
        y = torch.cat((y_0, y_1), dim=0)
        bat_id = np.random.rand(x.shape[0]).argsort()
        bat_id_rev = bat_id.argsort()
        x = x[bat_id, ]
        y = y[bat_id, ]
        # bat_id_rev = bat_id.argsort()
        # x_1 = x[bat_id, ].detach()
        # y_1 = y[bat_id, ].detach()
        # x_2 = x_1[bat_id_rev, ].detach()
        # y_2 = y_1[bat_id_rev, ].detach()
        # assert torch.all(x_2 == x)
        # assert torch.all(y_2 == y), '{} \n {}'.format(y_2, y)

        x = x.to(device)
        y = y.to(device)

        if args.squeeze_first:
            x = squeeze_layer(x)

        logits = model(torch.pixel_shuffle(x, 2))
        loss = criterion(logits, y)

        prd = logits.detach()
        prd = prd[bat_id_rev, ]
        lab = y.detach()
        lab = lab[bat_id_rev, ]
        compute_acc(prd[:prd.shape[0] // 2, ],
                    lab[:prd.shape[0] // 2, ],
                    met_0,
                    n_classes)
        compute_acc(prd[prd.shape[0] // 2:, ],
                    lab[prd.shape[0] // 2:, ],
                    met_1,
                    n_classes)

        loss.backward()
        optimizer.step()

    print_msg(logger, met_0, epoch, 'TRN_0', n_classes)
    print_msg(logger, met_1, epoch, 'TRN_1', n_classes)

    model.eval()
    met = np.zeros_like(best_trn_0)
    for _, (x, y) in enumerate(tst_loader[0]):
        x = x.to(device)
        y = y.to(device)

        if args.squeeze_first:
            x = squeeze_layer(x)
        lgts = model(torch.pixel_shuffle(x, 2))
        compute_acc(lgts, y, met, n_classes)

    print_msg(logger, met, epoch, 'TST', n_classes)

    is_best = best_tst[1, -1] / (best_tst[2, -1] + 1e-5) < \
        met[1, -1] / (met[2, -1] + 1e-5)
    if is_best:
        best_trn_0 = met_0
        best_trn_1 = met_1
        best_tst = met
        best_epoch = epoch
        model_file = save_path / 'best_model.pt'
        torch.save(model.state_dict(), str(model_file))

print_msg(logger, best_trn_0, best_epoch, 'TRN_0', n_classes, '[BEST] ')
print_msg(logger, best_trn_1, best_epoch, 'TRN_1', n_classes, '[BEST] ')
print_msg(logger, best_tst, best_epoch, 'TST', n_classes, '[BEST] ')
