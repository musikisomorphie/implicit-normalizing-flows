from numpy.core.arrayprint import BoolFormat
from lib.visualize_flow import visualize_transform
import lib.utils as utils
import lib.toy_data as toy_data
import lib.layers as layers
import lib.layers.base as base_layers
import lib.optimizers as optim
import torch
import numpy as np
import math
import time
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import pickle
matplotlib.use('Agg')


ACTIVATION_FNS = {
    'identity': base_layers.Identity,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
    'sin': base_layers.Sin,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)
parser.add_argument(
    '--arch', choices=['iresnet', 'realnvp', 'implicit'], default='implicit')
parser.add_argument('--coeff', type=float, default=0.9)
parser.add_argument('--vnorms', type=str, default='222222')
parser.add_argument('--n-lipschitz-iters', type=int, default=5)
parser.add_argument('--atol', type=float, default=None)
parser.add_argument('--rtol', type=float, default=None)
parser.add_argument('--learn-p', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--mixed', type=eval, choices=[True, False], default=True)

parser.add_argument('--dims', type=str, default='128-128-128-128')
parser.add_argument('--act', type=str,
                    choices=ACTIVATION_FNS.keys(), default='sin')
parser.add_argument('--nblocks', type=int, default=100)
parser.add_argument('--brute-force', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--actnorm', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--batchnorm', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--exact-trace', type=eval,
                    choices=[True, False], default=False)
parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument(
    '--n-dist', choices=['geometric', 'poisson'], default='geometric')

parser.add_argument('--nepochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--annealing-iters', type=int, default=0)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--save', type=str, default='experiments/iresnet_toy')
parser.add_argument('--viz_freq', type=int, default=1000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lganm-dir',
                    type=pathlib.Path,
                    default='/home/histopath/Data/Symm/',
                    metavar='DIR')
parser.add_argument('--exp',
                    choices=['fin', 'abcd'],
                    default='fin',
                    help='Experimental settings defined in AICP. (default: %(default)s)')
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(
    args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * np.log(2 * np.pi)
    return logZ - z.pow(2) / 2


def compute_loss(model, mask, dat, sym, beta=1.):
    env_size, batch_size = dat.shape[:2]

    x = dat[:, :, ~mask]
    x = x.reshape(-1, x.shape[-1])
    y_gt = dat[:, :, mask]

    # load data
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)
    y_gt = torch.from_numpy(y_gt).type(torch.float32).to(device)

    # transform to z
    flow, y_pd = model(x, zero)
    z, delta_logp = flow
    y_pd = y_pd.reshape(env_size, batch_size, 1)
    # compute log p(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - beta * delta_logp
    # loss = -torch.mean(logpx)
    loss = torch.mean(torch.abs(y_gt - y_pd))
    return loss, torch.mean(logpz), torch.mean(-delta_logp)


def parse_vnorms():
    ps = []
    for p in args.vnorms:
        if p == 'f':
            ps.append(float('inf'))
        else:
            ps.append(float(p))
    return ps[:-1], ps[1:]


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).mul(0.01).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


def build_nnet(dims, activation_fn=torch.nn.ReLU):
    nnet = []
    domains, codomains = parse_vnorms()
    if args.learn_p:
        if args.mixed:
            domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
        else:
            domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
        codomains = domains[1:] + [domains[0]]
    # print(dims, domains, codomains)
    for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
        # print(i, (in_dim, out_dim, domain, codomain))
        if i > 0:
            nnet.append(activation_fn())
        nnet.append(
            base_layers.get_linear(
                in_dim,
                out_dim,
                coeff=args.coeff,
                n_iterations=args.n_lipschitz_iters,
                atol=args.atol,
                rtol=args.rtol,
                domain=domain,
                codomain=codomain,
                zero_init=(out_dim == 2),
            )
        )
    return torch.nn.Sequential(*nnet)


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)


def get_ords(model):
    ords = []
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            domain, codomain = m.compute_domain_codomain()
            if torch.is_tensor(domain):
                domain = domain.item()
            if torch.is_tensor(codomain):
                codomain = codomain.item()
            ords.append(domain)
            ords.append(codomain)
    return ords


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'


def proc_lganm(lganm_dict,
               test_coef,
               dt_phase,
               dt_type,
               val_num):

    out_dict = dict()
    for phase in dt_phase:
        for typ in dt_type:
            out_dict[phase + '_' + typ] = list()

    for sym, env in lganm_dict['envs'].items():
        # convert the raw data to the format that feed to nn

        sym_id = sym[0] if sym[0] != -1 else lganm_dict['target']
        sym_np = np.zeros_like(env)
        sym_np[:, sym_id] = sym[-1]

        env_dict = {'env_' + dt_type[0]: env,
                    'env_' + dt_type[1]: sym_np}

        if sym[-1] != test_coef:
            phs = dt_phase[0]
        else:
            phs = dt_phase[2]

        for typ in dt_type:
            out_dict[phs + '_' +
                     typ].append(env_dict['env_' + typ][: -val_num, ])
            out_dict[dt_phase[1] + '_' +
                     typ].append(env_dict['env_' + typ][-val_num:, ])

    # print()
    for key, val in out_dict.items():
        # print(key)
        out_dict[key] = np.stack(val, axis=0)
        # print(key, out_dict[key].shape)
    # print()
    return out_dict


def shuffle_along_axis(dat, sym, axis):
    shuffle_idx = np.random.rand(*dat.shape).argsort(axis=axis)
    return np.take_along_axis(dat, shuffle_idx, axis=axis), \
        np.take_along_axis(sym, shuffle_idx, axis=axis)


def compute_acc(model, logger, epoch,
                mask, dat, sym, phase):

    metric = compute_loss(model, mask,
                          dat, sym)
    loss, logpz, delta_logp = metric
    log_message = ('[{}] Epoch {:04d} | Loss {:.6f} '
                   '| Logp(z) {:.6f} | DeltaLogp {:.6f}'.format(
                       phase.upper(), epoch,
                       loss.item(), logpz.item(), delta_logp.item()))
    logger.info(log_message)
    return loss


if __name__ == '__main__':
    # use interventional data as test data
    test_coef = 0.
    val_num = 128
    var_num = 12
    regress = True
    data_phase = ('trn', 'val', 'tst')
    data_type = ('dat', 'sym')

    pkl_dir = args.lganm_dir / args.exp / 'n_1000'
    for pkl_id, pkl_file in enumerate(pkl_dir.glob('*.pickle')):
        with open(str(pkl_file), 'rb') as pl:
            lganm_dict = pickle.load(pl)
            lganm_dict['truth'] = list(lganm_dict['truth'])
            dag_wei = lganm_dict['case'].sem.W
            assert (dag_wei.shape[1] == dag_wei.shape[0])
            assert np.all(np.asarray(lganm_dict['truth']) < dag_wei.shape[0])
            assert lganm_dict['target'] < dag_wei.shape[1]
            data_dict = proc_lganm(lganm_dict,
                                   test_coef,
                                   data_phase,
                                   data_type,
                                   val_num)

    activation_fn = ACTIVATION_FNS[args.act]

    dims = [var_num - 1] + list(map(int, args.dims.split('-'))) + [var_num - 1]
    if args.arch == 'iresnet':
        blocks = []
        if args.actnorm:
            blocks.append(layers.ActNorm1d(2))
        for _ in range(args.nblocks):
            blocks.append(
                layers.iResBlock(
                    build_nnet(dims, activation_fn),
                    n_dist=args.n_dist,
                    n_power_series=args.n_power_series,
                    exact_trace=args.exact_trace,
                    brute_force=args.brute_force,
                    n_samples=args.n_samples,
                    neumann_grad=False,
                    grad_in_forward=False,
                )
            )
            if args.actnorm:
                blocks.append(layers.ActNorm1d(2))
            if args.batchnorm:
                blocks.append(layers.MovingBatchNorm1d(2))
        model = layers.SequentialFlowToy(blocks, dims[0]).to(device)
    elif args.arch == 'implicit':
        # dims = [2] + list(map(int, args.dims.split('-'))) + [2]
        blocks = []
        if args.actnorm:
            blocks.append(layers.ActNorm1d(2))
        for _ in range(args.nblocks):
            blocks.append(
                layers.imBlock(
                    build_nnet(dims, activation_fn),
                    build_nnet(dims, activation_fn),
                    n_dist=args.n_dist,
                    n_power_series=args.n_power_series,
                    exact_trace=args.exact_trace,
                    brute_force=args.brute_force,
                    n_samples=args.n_samples,
                    neumann_grad=False,
                    grad_in_forward=False,  # toy data needn't save memory
                )
            )
        model = layers.SequentialFlowToy(blocks, dims[0]).to(device)
    elif args.arch == 'realnvp':
        blocks = []
        for _ in range(args.nblocks):
            blocks.append(layers.CouplingBlock(2, swap=False))
            blocks.append(layers.CouplingBlock(2, swap=True))
            if args.actnorm:
                blocks.append(layers.ActNorm1d(2))
            if args.batchnorm:
                blocks.append(layers.MovingBatchNorm1d(2))
        model = layers.SequentialFlowToy(blocks, dims[0]).to(device)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(
        count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    logpz_meter = utils.RunningAverageMeter(0.93)
    delta_logp_meter = utils.RunningAverageMeter(0.93)

    end = time.time()

    # if (args.resume is not None):
    #     logger.info('Resuming model from {}'.format(args.resume))
    #     with torch.no_grad():
    #         x = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    #         x = torch.from_numpy(x).type(torch.float32).to(device)
    #         model(x, restore=True)
    #     checkpt = torch.load(args.resume)
    #     sd = {k: v for k, v in checkpt['state_dict'].items(
    #     ) if 'last_n_samples' not in k}
    #     state = model.state_dict()
    #     state.update(sd)
    #     model.load_state_dict(state, strict=True)
    #     del checkpt
    #     del state
    # else:
    #     with torch.no_grad():
    #         x = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
    #         x = torch.from_numpy(x).type(torch.float32).to(device)
    #         model(x, restore=True)

    mask = np.zeros(var_num, dtype=bool)
    mask[lganm_dict['truth']] = True
    best_loss = float('inf')
    tot_iter = 1
    model.train()
    for epoch in range(args.nepochs):

        trn = shuffle_along_axis(data_dict['trn_dat'],
                                 data_dict['trn_sym'],
                                 axis=1)
        trn_dat, trn_sym = trn
        trn_iter = trn_dat.shape[1] // args.batch_size
        if trn_dat.shape[1] % args.batch_size != 0:
            trn_iter += 1

        for iter in range(trn_iter):
            optimizer.zero_grad()
            cur_dat = trn_dat[:,
                              iter * args.batch_size: (iter + 1) * args.batch_size,
                              :]
            cur_sym = trn_sym[:,
                              iter * args.batch_size: (iter + 1) * args.batch_size,
                              :]
            beta = min(1, tot_iter /
                       args.annealing_iters) if args.annealing_iters > 0 else 1.
            loss, logpz, delta_logp = compute_loss(model,
                                                   mask,
                                                   cur_dat,
                                                   cur_sym,
                                                   beta=beta)
            loss_meter.update(loss.item())
            logpz_meter.update(logpz.item())
            delta_logp_meter.update(delta_logp.item())
            loss.backward()
            if args.learn_p and epoch > args.annealing_iters:
                compute_p_grads(model)
            optimizer.step()
            update_lipschitz(model, args.n_lipschitz_iters)
            tot_iter += 1

            time_meter.update(time.time() - end)

        # if itr % args.log_freq == 0:
        logger.info('[TRN] Epoch {:04d} | Loss {:.6f}({:.6f})'
                    ' | Logp(z) {:.6f}({:.6f}) | DeltaLogp {:.6f}({:.6f}) | Time {:.4f}({:.4f})'.format(
                        epoch, loss_meter.val, loss_meter.avg, logpz_meter.val, logpz_meter.avg,
                        delta_logp_meter.val, delta_logp_meter.avg, time_meter.val, time_meter.avg))

        # if itr % args.val_freq == 0 or itr == args.niters:
        update_lipschitz(model, 200)
        with torch.no_grad():
            model.eval()
            val_loss = compute_acc(model, logger, epoch, mask,
                                   data_dict['val_dat'],
                                   data_dict['val_sym'],
                                   'VAL')

            # logger.info('Ords: {}'.format(pretty_repr(get_ords(model))))

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                utils.makedirs(args.save)
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, os.path.join(args.save, 'checkpt.pth'))

            tst_loss = compute_acc(model, logger, epoch, mask,
                                   data_dict['tst_dat'],
                                   data_dict['tst_sym'],
                                   'TST')

            model.train()
        print()

        # if itr == 1 or itr % args.viz_freq == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         p_samples = toy_data.inf_train_gen(args.data, batch_size=20000)

        #         sample_fn, density_fn = model.module.inverse, model.forward

        #         plt.figure(figsize=(9, 3))
        #         visualize_transform(
        #             p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
        #             samples=True, npts=400, device=device
        #         )
        #         fig_filename = os.path.join(
        #             args.save, 'figs', '{:04d}.jpg'.format(itr))
        #         utils.makedirs(os.path.dirname(fig_filename))
        #         plt.savefig(fig_filename)
        #         plt.close()
        #         model.train()

        end = time.time()

    logger.info('Training has finished.')
