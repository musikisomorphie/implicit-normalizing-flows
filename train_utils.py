import sys
import math
import numpy as np
import pathlib
import logging
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from skimage import color
from numbers import Number

from lib.implicit_flow import ImplicitFlow
from lib.resflow import ResidualFlow
import lib.datasets as datasets
import lib.layers as layers
import lib.layers.base as base_layers

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]

    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            # transforms.Resize([128, 128]),
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # t_standardize,
        ]
    else:
        transforms_ls = [
            # transforms.Resize([128, 128]),
            transforms.ToTensor(),
            # t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform


def initialize_scrc_transform(is_training):
    angles = [0, 90, 180, 270]

    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
        ]
        transform = transforms.Compose(transforms_ls)
        return transform
    else:
        return None


class PredNet(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes,
                 chn_dim,
                 use_pretrained=True):
        super(PredNet, self).__init__()

        model_name = model_name.lower()
        if model_name == "resnet":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            model_ft.conv1 = nn.Conv2d(chn_dim, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == "densenet":
            """ Densenet121
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            model_ft.conv0 = nn.Conv2d(chn_dim, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        elif model_name == "mobilenet":
            """ mobilenet v2
            """
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

        elif model_name == "shufflenet":
            """ shufflenet v2
            """
            model_ft = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        self.backbone = model_ft

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3)).detach().clone()
        std = x.std(dim=(2, 3)).detach().clone()
        std[std == 0.] = 1.
        # mean = torch.as_tensor(mean).to(x)
        # std = torch.as_tensor(std).to(x)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        std = std.unsqueeze(-1).unsqueeze(-1)
        # if (std == 0).any():
        #     raise ValueError(
        #         'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        # if mean.ndim == 1:
        #     mean = mean.view(-1, 1, 1)
        # if std.ndim == 1:
        #     std = std.view(-1, 1, 1)
        # x.sub_(mean).div_(std)
        # print(x.shape, mean.shape, std.shape)
        return (x - mean) / std

    def forward(self, x):
        x = self.standardize(x)
        x = self.backbone(x)
        return x


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.module = module
        self.decay = decay
        self.shadow_params = {}
        self.nparams = sum(p.numel() for p in module.parameters())

    def init(self):
        for name, param in self.module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def apply(self):
        if len(self.shadow_params) == 0:
            self.init()
        else:
            with torch.no_grad():
                for name, param in self.module.named_parameters():
                    self.shadow_params[name] -= (1 - self.decay) * \
                        (self.shadow_params[name] - param.data)

    def set(self, other_ema):
        self.init()
        with torch.no_grad():
            for name, param in other_ema.shadow_params.items():
                self.shadow_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module.named_parameters():
            param.data.copy_(self.shadow_params[name])

    def swap(self):
        for name, param in self.module.named_parameters():
            tmp = self.shadow_params[name].clone()
            self.shadow_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HEDJitter(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """

    # HED_light: theta=0.05; HED_strong: theta=0.2
    def __init__(self, theta=0., keep_hed=False):
        assert isinstance(
            theta, Number), "theta should be a single number."
        self.theta = theta
        self.keep_hed = keep_hed

    @staticmethod
    def adjust_HED(image, theta, keep_hed):
        alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        betti = np.random.uniform(-theta, theta, (1, 3))

        assert image.shape[0] in (3, 4, 5)
        if image.shape[0] > 3:
            nul = image[3:, ].clone().numpy()
        img = image[:3, ].clone().permute(1, 2, 0)
        img = img.numpy()
        assert img.shape[-1] == 3

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space

        nimg = np.reshape(ns, img.shape)
        if not keep_hed:
            nimg = color.hed2rgb(nimg)

        # # pt data visualization
        # rand_id = random.randint(0, 10000)
        # imin = nimg.min()
        # imax = nimg.max()
        # pt_path = '/raid/jiqing/Data/SCRC_visual/visual_pt/'

        # rsimg = (255 * (nimg - imin) / (imax - imin))
        # rsimg_out = Image.fromarray(np.uint8(rsimg))
        # rsimg_out.save(pt_path + '{}_aug.png'.format(rand_id), 'PNG')
        # orimg = (255 * img)
        # orimg_out = Image.fromarray(np.uint8(orimg))
        # orimg_out.save(pt_path + '{}_org.png'.format(rand_id), 'PNG')

        # rsimg_vis = visual_instances(inst_np=nul[0, :, :].copy(),
        #                              cell_color=NUL_CLR,
        #                              inst_type=nul[1, :, :].copy(),
        #                              canvas=rsimg.copy())
        # rsimg_vis = Image.fromarray(np.uint8(rsimg_vis))
        # rsimg_vis.save(pt_path + '{}_auv.png'.format(rand_id), 'PNG')
        # orimg_vis = visual_instances(inst_np=nul[0, :, :].copy(),
        #                              cell_color=NUL_CLR,
        #                              inst_type=nul[1, :, :].copy(),
        #                              canvas=orimg.copy())
        # orimg_vis = Image.fromarray(np.uint8(orimg_vis))
        # orimg_vis.save(pt_path + '{}_orv.png'.format(rand_id), 'PNG')

        nimg = torch.from_numpy(nimg).permute(2, 0, 1)
        image[:3, ] = nimg
        return image

    def __call__(self, img):
        return self.adjust_HED(img, self.theta, self.keep_hed)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0})'.format(self.theta)
        # format_string += ',alpha={0}'.format(self.alpha)
        # format_string += ',betti={0}'.format(self.betti)
        return format_string


class ResizeMix(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """

    def __init__(self, size, near_dim=None):
        if not isinstance(size, (int, list, tuple)):
            raise TypeError(
                "Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, (list, tuple)) and len(size) != 2:
            raise ValueError(
                "If size is a sequence, it should have 2 values")
        self.size = size

        if (not isinstance(near_dim, (int, list, tuple))) and \
                (near_dim is not None):
            raise TypeError(
                "The dimension along which nearest interpolation applied should be int or sequence. Got {}".format(type(near_dim)))
        if isinstance(near_dim, (list, tuple)) and \
                (not all(isinstance(dim, int) for dim in near_dim)):
            raise TypeError(
                "the value in near_dim should all be integer.")
        self.near_dim = near_dim

    def __call__(self, img):
        if self.near_dim is None:
            out = TF.resize(img, self.size)
        else:
            lab_msk = torch.zeros(img.shape[0], dtype=torch.bool)
            lab_msk[self.near_dim] = True
            # print(lab_msk)
            out = torch.zeros([img.shape[0],
                               self.size[0],
                               self.size[1]]).to(img)
            out[lab_msk] = TF.resize(img[lab_msk],
                                     self.size,
                                     TF.InterpolationMode.NEAREST)
            out[~lab_msk] = TF.resize(img[~lab_msk],
                                      self.size,
                                      TF.InterpolationMode.NEAREST)
        return out

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'near_dim={})'.format(self.near_dim)
        return format_string


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(str(logpath), mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(str(filepath))
    with open(str(filepath), "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
            lipschitz_constants.append(m.scale)
    return lipschitz_constants


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


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


def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    norm_logZ = logZ - z.pow(2) / 2
    return norm_logZ.view(z.shape[0], -1)


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_bits(x, nbits):
    if nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - nbits))
        x = x / 2**nbits
    return x


def add_noise(x, add_noise, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def update_lr(optimizer, itr, warmup_iters, lr):
    iter_frac = min(float(itr + 1) / max(warmup_iters, 1), 1.0)
    lr *= iter_frac
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def add_padding(x, padding, padding_dist, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if padding > 0:
        if padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], padding,
                            x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], padding, x.shape[2], x.shape[3]).normal_(
                nvals / 2, nvals / 8)
            logpu = normal_logprob(
                u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x, padding, im_dim):
    if padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x


def parallelize(model):
    return torch.nn.DataParallel(model)


def append_cms(x, y, n_class):
    # print(x.shape, y.shape)
    x_append = y.view(y.shape[0], 1, 1, 1) * torch.ones(
        (x.shape[0], 1, x.shape[2], x.shape[3])).to(x) / n_class
    return torch.cat((x, x_append), dim=1)


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.imBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


def initialize_model(model_name,
                     num_classes,
                     chn_dim,
                     use_pretrained=False):
    """Select the model with `model name'

    Args:
        num_classes: number of class
        num_pred: number of prediction data
        use_pretrained: True: load pretrained model, False: not load

    Returns:
        the initialized model
    """

    model_ft = None
    model_name = model_name.lower()
    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft.conv1 = nn.Conv2d(chn_dim, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft.conv0 = nn.Conv2d(chn_dim, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "mobilenet":
        """ mobilenet v2
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "shufflenet":
        """ shufflenet v2
        """
        model_ft = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError('Invalid model name {}'.format(model_name))

    return model_ft


def data_prep(args):
    if args.dataset == 'rxrx1':
        trn_trans = initialize_rxrx1_transform(True)
        eval_trans = initialize_rxrx1_transform(False)
        dataset_kwargs = dict()
        split_scheme = 'official'
        num_chn = 3
        eval_data = ['val', 'test', 'id_test']
    elif args.dataset == 'scrc':
        trn_trans = initialize_scrc_transform(True)
        eval_trans = initialize_scrc_transform(False)
        if args.inp == 'i':
            img_chn = [1, 2, 3]
        elif args.inp == 'mi':
            img_chn = [0, 1, 2, 3]
        elif args.inp == 'm':
            img_chn = [0]
        dataset_kwargs = {'img_chn': img_chn}
        split_scheme = args.env
        num_chn = len(img_chn)
        eval_data = ['val', 'test']

    data_loader = []
    loader_kwargs = {'num_workers': args.nworkers}
    data = get_dataset(dataset=args.dataset,
                       root_dir=pathlib.Path(args.dataroot),
                       split_scheme=split_scheme,
                       **dataset_kwargs)

    trn_data = data.get_subset('train',
                               transform=trn_trans)
    data_loader.append(get_train_loader('standard',
                                        trn_data,
                                        batch_size=args.batchsize,
                                        **loader_kwargs))

    for evl in eval_data:
        sub_data = data.get_subset(evl,
                                   transform=eval_trans)
        data_loader.append(get_eval_loader('standard',
                                           sub_data,
                                           batch_size=args.eval_batchsize))

    input_size = [args.batchsize,
                  num_chn,
                  args.imagesize,
                  args.imagesize]

    print('rxrx1', data.n_classes)
    return data_loader, data.n_classes, input_size


def normflow(args, input_size):
    if args.flow == 'imflow':
        norm_flow = ImplicitFlow
    elif args.flow == 'reflow':
        norm_flow = ResidualFlow

    if args.dataset == 'scrc' and args.inp == 'mi':
        _left_pad = args.shuffle_factor ** 2
    else:
        _left_pad = 0

    model = norm_flow(
        input_size=input_size,
        scale_factor=args.scale_factor,
        shuffle_factor=args.shuffle_factor,
        couple_label=args.couple_label,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim,
        factor_out=args.factor_out,
        quadratic=args.quadratic,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        vnorms=args.vnorms,
        n_lipschitz_iters=args.n_lipschitz_iters,
        sn_atol=args.sn_tol,
        sn_rtol=args.sn_tol,
        n_power_series=args.n_power_series,
        n_dist=args.n_dist,
        n_samples=args.n_samples,
        kernels=args.kernels,
        activation_fn=args.act,
        fc_end=args.fc_end,
        fc_idim=args.fc_idim,
        n_exact_terms=args.n_exact_terms,
        preact=args.preact,
        neumann_grad=args.neumann_grad,
        grad_in_forward=args.mem_eff,
        first_resblock=True,
        learn_p=args.learn_p,
        left_pad=_left_pad)

    return model


def custom_logger(logger_name, level=logging.DEBUG):
    """Method to return a custom logger with the given name and level
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                     "%(lineno)d — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def test(aa=0, **kwargs):
    print('{}'.format(kwargs['bb']))


def main():
    kwargs = {'bb': 1}
    test(**kwargs)


if __name__ == '__main__':
    main()
