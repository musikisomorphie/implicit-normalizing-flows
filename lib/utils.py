import os
import math
import torch.nn as nn
from torchvision import models
from numbers import Number
import logging
import torch
import numpy as np
from skimage import color
from PIL import Image
import random
import cv2

NUL_CLR = {1: [0, 0, 255],  # blue, 16711680
           2: [255, 0, 0],  # red, 255
           3: [255, 0, 255],  # magenta, 16711935
           4: [0, 128, 0],  # dark green, 32768
           5: [0, 255, 255]}  # cyan, 16776960


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


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


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch, last_checkpoints=None, num_checkpoints=None):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

    if last_checkpoints is not None and num_checkpoints is not None:
        last_checkpoints.append(epoch)
        if len(last_checkpoints) > num_checkpoints:
            rm_epoch = last_checkpoints.pop(0)
            os.remove(os.path.join(save, 'checkpt-%04d.pth' % rm_epoch))


def isnan(tensor):
    return (tensor != tensor)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


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

    def __init__(self, theta=0.):  # HED_light: theta=0.05; HED_strong: theta=0.2
        assert isinstance(
            theta, Number), "theta should be a single number."
        self.theta = theta

    @staticmethod
    def adjust_HED(image, theta):
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
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        # pt data visualization
        rand_id = random.randint(0, 10000)
        imin = nimg.min()
        imax = nimg.max()
        pt_path = '/raid/jiqing/Data/SCRC_visual/visual_pt/'

        rsimg = (255 * (nimg - imin) / (imax - imin))
        rsimg_out = Image.fromarray(np.uint8(rsimg))
        rsimg_out.save(pt_path + '{}_aug.png'.format(rand_id), 'PNG')
        orimg = (255 * img)
        orimg_out = Image.fromarray(np.uint8(orimg))
        orimg_out.save(pt_path + '{}_org.png'.format(rand_id), 'PNG')

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
        return self.adjust_HED(img, self.theta)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0})'.format(self.theta)
        # format_string += ',alpha={0}'.format(self.alpha)
        # format_string += ',betti={0}'.format(self.betti)
        return format_string


def bbox(img):
    """Compute the bbox coordinates of an object embbeded 
    in an img

    Args:
        img: the boolean array with True at the pos of an object
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def visual_instances(inst_np,
                     cell_color,
                     inst_type,
                     canvas=None):
    """Draw the cell masks on an image array

    Args:
        inst_np: the array recording cell instances
        cell_color: the list of colors related to cell_inst
        cell_inst: the list of cell instances 
        canvas: the output image with colored masks
    """

    canvas = np.full(inst_np.shape + (3,), 0., dtype=np.float) \
        if canvas is None else np.copy(canvas)

    for idx, inst_id in enumerate(np.unique(inst_np).tolist()):
        if inst_id == 0:
            continue
        inst_map = np.array(inst_np == inst_id, np.uint8)
        # if the inst_id does not exist
        if not np.any(inst_map):
            print(('the cell {} polygon does not exist, could be '
                   'overwritten by other cells recorded later.').format(inst_id))
            continue

        clr_idx = np.unique(inst_type[inst_np == inst_id]).tolist()
        if len(clr_idx) != 1 or clr_idx[0] == 0:
            print(('the cell type {} is illegal, thus ignore.').format(clr_idx))
            continue

        # print(idx, inst_id, cell_color[idx])
        y1, y2, x1, x2 = bbox(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_np.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_np.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours, _ = cv2.findContours(inst_map_crop,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        rgb = cell_color[int(clr_idx[0])]
        cv2.drawContours(inst_canvas_crop,
                         contours, -1,
                         rgb, 1)

        canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas
