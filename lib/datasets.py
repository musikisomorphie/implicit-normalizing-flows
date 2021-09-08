import torch
import torchvision.datasets as vdsets
from torch.utils.data.dataset import Dataset


class CustomData(object):

    def __init__(self, loc, transform=None, in_mem=True):
        self.in_mem = in_mem
        self.dataset = torch.load(loc)
        if in_mem:
            self.dataset = self.dataset.float().div(255)
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        if not self.in_mem:
            x = x.float().div(255)
        x = self.transform(x) if self.transform is not None else x
        return x, 0


class MNIST(object):

    def __init__(self, dataroot, train=True, transform=None):
        self.mnist = vdsets.MNIST(
            dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, index):
        return self.mnist[index]


class CIFAR10(object):

    def __init__(self, dataroot, train=True, transform=None):
        self.cifar10 = vdsets.CIFAR10(
            dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar10)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, index):
        return self.cifar10[index]


class CelebA5bit(object):

    LOC = 'data/celebahq64_5bit/celeba_full_64x64_5bit.pth'

    def __init__(self, train=True, transform=None):
        self.dataset = torch.load(self.LOC).float().div(31)
        if not train:
            self.dataset = self.dataset[:5000]
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        x = self.transform(x) if self.transform is not None else x
        return x, 0


class CelebAHQ(CustomData):
    TRAIN_LOC = 'data/celebahq/celeba256_train.pth'
    TEST_LOC = 'data/celebahq/celeba256_validation.pth'

    def __init__(self, train=True, transform=None):
        return super(CelebAHQ, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform)


class Imagenet32(CustomData):
    TRAIN_LOC = 'data/imagenet32/train_32x32.pth'
    TEST_LOC = 'data/imagenet32/valid_32x32.pth'

    def __init__(self, train=True, transform=None):
        return super(Imagenet32, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform)


class Imagenet64(CustomData):
    TRAIN_LOC = 'data/imagenet64/train_64x64.pth'
    TEST_LOC = 'data/imagenet64/valid_64x64.pth'

    def __init__(self, train=True, transform=None):
        return super(Imagenet64, self).__init__(self.TRAIN_LOC if train else self.TEST_LOC, transform, in_mem=False)


class SCRC(Dataset):
    def __init__(self,
                 scale_factor,
                 n_classes,
                 couple_label,
                 scrc_path,
                 scrc_pat=False,
                 scrc_idx=None,
                 scrc_in=None,
                 scrc_out=None,
                 transforms=None):
        self.scale_factor = scale_factor
        self.n_classes = n_classes
        self.couple_label = couple_label
        self.scrc_pat = scrc_pat

        imgs, labs = torch.load(str(scrc_path))
        self.n_nuclei = torch.amax(imgs[:, -1])
        print('nuclei classes {}'.format(self.n_nuclei))
        if scrc_idx is not None:
            imgs = imgs[scrc_idx]
            labs = labs[scrc_idx]

        self._proc_img(imgs, scrc_in)
        self._proc_lab(labs, scrc_out)
        assert self.imgs.shape[0] == self.labs.shape[0]

        self.len, self.chn = list(self.imgs.shape[:2])
        self.transforms = transforms

    def _proc_img(self, imgs, scrc_in=None):
        # for i in range(5):
        #     print(torch.amin(imgs[:, i, :, :]),
        #           torch.amax(imgs[:, i, :, :]))

        self.imgs = imgs.float()
        # divide the rgb pixel by 255
        self.imgs[:, :3] = self.imgs[:, :3].div(255.)
        # divide the nuclei type by nuclei classes
        self.imgs[:, -1] = self.imgs[:, -1].div(self.n_nuclei)
        if scrc_in is not None:
            self.imgs = self.imgs[:, scrc_in]

        if [0, 1, 2, 4] == scrc_in:
            print('switch to mask_dim=0, del mucin=5 and misc=3')
            self.imgs = self.imgs[:, [-1, 0, 1, 2]]
            _ncls = self.imgs[:, 0]
            _ncls_idx = (_ncls == 3 / self.n_nuclei) | \
                (_ncls == 5 / self.n_nuclei)
            _ncls[_ncls_idx] = 0.
            self.imgs[:, 0] = _ncls
            self.imgs[:, 1:] += _ncls.unsqueeze(1)
            print(torch.unique(self.imgs[:, 0]))

    def _proc_lab(self, labs, scrc_out=None):
        if scrc_out is not None:
            if not isinstance(scrc_out, str):
                raise TypeError('The outcome {} is not a string.'.
                                format(scrc_out))
            scrc_out = scrc_out.lower()
            if scrc_out == 'os':
                self.labs = labs[:, -8:-6].float()
            elif scrc_out == 'dfs':
                self.labs = labs[:, -6:-4].float()
            elif scrc_out == 'cms':
                self.labs = labs[:, -4:]
                self.labs = torch.argmax(self.labs, dim=1)
                print(torch.amin(self.labs), torch.amax(
                    self.labs), self.labs.shape)
            else:
                raise ValueError('The outcome {} is not cms, os or dfs.'.
                                 format(scrc_out))
        else:
            self.labs = labs.float()

        if self.scrc_pat:
            self.pat = labs[:, 0]

    def __getitem__(self, index):
        img = self.imgs[index]
        lab = self.labs[index]
        if self.scrc_pat:
            pat = self.pat[index]

        if self.transforms is not None:
            img = self.transforms(img)

        img = torch.pixel_unshuffle(img, self.scale_factor)
        if self.couple_label:
            lab_out = lab.float() * torch.ones((1,
                                                img.shape[1],
                                                img.shape[2])) / self.n_classes
            img = torch.cat((lab_out, img))
            lab = lab.long()

        if self.scrc_pat:
            # print(lab.shape, pat.shape)
            # print(lab, pat)
            lab = torch.stack((lab, pat)).long()

        return img, lab

    @property
    def ndim(self):
        return self.chn

    def __len__(self):
        return self.len
