import numpy as np
import torch
import torch.nn as nn
import train_utils as utils
import lib.layers as layers
import lib.layers.base as base_layers

ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: nn.ReLU(inplace=b),
    'sin': lambda b: base_layers.Sin(),
    'zero': lambda b: base_layers.Zero(),
    'zeropad': lambda b: base_layers.PadZero(pad_dim=b),
}


class ResidualFlow(nn.Module):

    def __init__(
        self,
        classifier,
        couple_label,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        factor_out=True,
        quadratic=False,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=True,
        learn_p=False,
        classification=False,
        classification_hdim=64,
        block_type='resblock',
    ):
        super(ResidualFlow, self).__init__()
        self.n_scale = min(len(n_blocks), self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.factor_out = factor_out
        self.quadratic = quadratic
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.vnorms = vnorms
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.kernels = kernels
        self.activation_fn = activation_fn
        self.fc_end = fc_end
        self.fc_idim = fc_idim
        self.n_exact_terms = n_exact_terms
        self.preact = preact
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.first_resblock = first_resblock
        self.learn_p = learn_p
        self.classification = classification
        self.classification_hdim = classification_hdim
        self.block_type = block_type

        if not self.n_scale > 0:
            raise ValueError(
                'Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

        if self.classification:
            self.classifier = classifier

        self.couple_label = couple_label

        self.fixed_z = utils.standard_normal_sample(
            [input_size[0] * 2, *input_size[1:]])

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        _stacked_blocks = StackediResBlocks if self.block_type == 'resblock' else StackedCouplingBlocks
        for i in range(self.n_scale):
            transforms.append(
                _stacked_blocks(
                    initial_size=(c, h, w),
                    idim=self.intermediate_dim,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    quadratic=self.quadratic,
                    actnorm=self.actnorm,
                    fc_actnorm=self.fc_actnorm,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    fc=self.fc,
                    coeff=self.coeff,
                    vnorms=self.vnorms,
                    n_lipschitz_iters=self.n_lipschitz_iters,
                    sn_atol=self.sn_atol,
                    sn_rtol=self.sn_rtol,
                    n_power_series=self.n_power_series,
                    n_dist=self.n_dist,
                    n_samples=self.n_samples,
                    kernels=self.kernels,
                    activation_fn=self.activation_fn,
                    fc_end=self.fc_end,
                    fc_idim=self.fc_idim,
                    n_exact_terms=self.n_exact_terms,
                    preact=self.preact,
                    neumann_grad=self.neumann_grad,
                    grad_in_forward=self.grad_in_forward,
                    first_resblock=self.first_resblock and (i == 0),
                    learn_p=self.learn_p,
                )
            )
            c, h, w = c * 2 if self.factor_out else c * 4, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 4**k, h // 2**k, w // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    # def build_classifier(self, chn_dim):
    #     self.classifier = utils.initialize_model(self.model_name,
    #                                              self.n_classes,
    #                                              chn_dim)

    def forward(self, x, logpx=None, inverse=False, classify=False, restore=False):
        if inverse:
            assert torch.is_tensor(x)
            if len(x.shape) == 1:
                print('fake shape', x.shape)
                x = self.fixed_z[:x.shape[0]].to(x)
            elif x.shape[1] < self.fixed_z.shape[1]:
                print('nucl shape', x.shape, self.fixed_z.shape)
                assert self.fixed_z.shape[0] >= x.shape[0]
                x = torch.cat((self.fixed_z[:x.shape[0], :-x.shape[1]].to(x),
                               x), dim=1)
            #     fixed_z = input.view(-1, *self.trans_size[1:])
            # else:
            #     assert input.shape[0] == 1
            #     if input:
            #         fixed_z = self.fixed_z
            #     else:
            #         fixed_z = utils.standard_normal_sample(self.trans_size)

            # out = self.inverse(self.fixed_z.to(input), None)
            # out = torch.pixel_shuffle(out[:, :-1], self.scale_factor)
            # return out
            return self.inverse(x, None)

        if classify:
            if self.couple_label:
                logits = self.classifier(x[:, :-1])
            else:
                logits = self.classifier(x)

        out = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                x, logpx = self.transforms[idx].forward(
                    x, logpx, restore=restore)
            else:
                x = self.transforms[idx].forward(x, restore=restore)
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                f, x = x[:, :d], x[:, d:]
                out.append(f)

        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, logpx)
        if classify:
            return output, logits
        else:
            return output

    def inverse(self, z, logpz=None):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i:i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize)
                  for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].inverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((zs[idx], z_prev), dim=1)
                    z_prev = self.transforms[idx].inverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].inverse(zs[-1], logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((zs[idx], z_prev), dim=1)
                    z_prev, logpz = self.transforms[idx].inverse(z_prev, logpz)
                return z_prev, logpz
        else:
            z = z.view(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].inverse(z)
                else:
                    z, logpz = self.transforms[idx].inverse(z, logpz)
            return z if logpz is None else (z, logpz)


class StackediResBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=True,
        learn_p=False,
        zero_pad=4
    ):

        chain = []

        # Parse vnorms
        ps = []
        for p in vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        domains, codomains = ps[:-1], ps[1:]
        assert len(domains) == len(kernels.split('-'))

        def _lipschitz_layer(fc):
            return base_layers.get_linear if fc else base_layers.get_conv2d

        def _actnorm(size, fc, zero_pad=0):
            # TODO fc is not compatiable with zero_pad
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0], zero_pad)

        def _quadratic_layer(initial_size, fc, zero_pad=0):
            # TODO fc is not compatible with zero_grad
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _resblock(initial_size, fc, idim, first_resblock, n_block, zero_pad=0):
            # TODO fc is not compatible with zero_grad
            if fc:
                return layers.iResBlock(
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_lipschitz_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        coeff=coeff,
                        domains=domains,
                        codomains=codomains,
                        n_iterations=n_lipschitz_iters,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        sn_atol=sn_atol,
                        sn_rtol=sn_rtol,
                        learn_p=learn_p,
                    ),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )
            else:
                def build_nnet():
                    ks = list(map(int, kernels.split('-')))
                    if learn_p:
                        _domains = [nn.Parameter(torch.tensor(0.))
                                    for _ in range(len(ks))]
                        _codomains = _domains[1:] + [_domains[0]]
                    else:
                        _domains = domains
                        _codomains = codomains
                    nnet = []
                    if not (first_resblock and n_block == 0) and preact:
                        if batchnorm:
                            nnet.append(
                                layers.MovingBatchNorm2d(initial_size[0]))
                        nnet.append(ACT_FNS[activation_fn](False))
                    nnet.append(
                        _lipschitz_layer(fc)(
                            initial_size[0], idim, ks[0], 1, ks[0] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                            domain=_domains[0], codomain=_codomains[0], atol=sn_atol, rtol=sn_rtol
                        )
                    )
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                    for i, k in enumerate(ks[1:-1]):
                        nnet.append(
                            _lipschitz_layer(fc)(
                                idim, idim, k, 1, k // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                                domain=_domains[i + 1], codomain=_codomains[i +
                                                                            1], atol=sn_atol, rtol=sn_rtol
                            )
                        )
                        if batchnorm:
                            nnet.append(layers.MovingBatchNorm2d(idim))
                        nnet.append(ACT_FNS[activation_fn](True))
                    if dropout:
                        nnet.append(nn.Dropout2d(dropout, inplace=True))

                    nnet.append(
                        _lipschitz_layer(fc)(
                            idim, initial_size[0] - zero_pad, ks[-1], 1, ks[-1] // 2, coeff=coeff, n_iterations=n_lipschitz_iters,
                            domain=_domains[-1], codomain=_codomains[-1], atol=sn_atol, rtol=sn_rtol
                        )
                    )
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(
                            initial_size[0] - zero_pad))

                    if zero_pad:
                        nnet.append(ACT_FNS['zeropad'](zero_pad))
                    return nn.Sequential(*nnet)
                return layers.iResBlock(
                    build_nnet(),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )

        if not first_resblock and init_layer is not None:
            chain.append(init_layer)
        if first_resblock and actnorm:
            chain.append(_actnorm(initial_size, fc, zero_pad))
        if first_resblock and fc_actnorm:
            chain.append(_actnorm(initial_size, True, zero_pad))

        if squeeze:
            c, h, w = initial_size
            z_pad = zero_pad if first_resblock else 0
            for i in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, idim,
                                       first_resblock, i, z_pad))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc, z_pad))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True, z_pad))
            chain.append(layers.SqueezeLayer(2))
        else:
            # TODO: last layer also need to append zero
            for i in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, idim,
                                       first_resblock, i))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim,
                                           first_resblock, n_blocks))
                    if actnorm or fc_actnorm:
                        chain.append(_actnorm(initial_size, True))

        super(StackediResBlocks, self).__init__(chain)


class FCNet(nn.Module):

    def __init__(
        self, input_shape, idim, lipschitz_layer, nhidden, coeff, domains, codomains, n_iterations, activation_fn,
        preact, dropout, sn_atol, sn_rtol, learn_p, div_in=1
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        dim = c * h * w
        nnet = []
        last_dim = dim // div_in
        if preact:
            nnet.append(ACT_FNS[activation_fn](False))
        if learn_p:
            domains = [nn.Parameter(torch.tensor(0.))
                       for _ in range(len(domains))]
            codomains = domains[1:] + [domains[0]]
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                    last_dim, idim, coeff=coeff, n_iterations=n_iterations, domain=domains[
                        i], codomain=codomains[i],
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout:
            nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                last_dim, dim, coeff=coeff, n_iterations=n_iterations, domain=domains[-1], codomain=codomains[-1],
                atol=sn_atol, rtol=sn_rtol
            )
        )
        self.nnet = nn.Sequential(*nnet)

    def forward(self, x, restore=False):
        x = x.view(x.shape[0], -1)
        y = self.nnet(x)
        return y.view(y.shape[0], *self.input_shape)


class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, logpx=None, restore=False):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logpx is None:
            y = self.fc_module(x)
            return y.view(*shape)
        else:
            y, logpy = self.fc_module(x, logpx)
            return y.view(*shape), logpy

    def inverse(self, y, logpy=None):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logpy is None:
            x = self.fc_module.inverse(y)
            return x.view(*shape)
        else:
            x, logpx = self.fc_module.inverse(y, logpy)
            return x.view(*shape), logpx


class StackedCouplingBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        # yapf: disable
        class nonloc_scope:
            pass
        nonloc_scope.swap = True
        # yapf: enable

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _weight_layer(fc):
            return nn.Linear if fc else nn.Conv2d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                nonloc_scope.swap = not nonloc_scope.swap
                return layers.CouplingBlock(
                    initial_size[0],
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_weight_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        coeff=None,
                        domains=None,
                        codomains=None,
                        n_iterations=None,
                        sn_atol=None,
                        sn_rtol=None,
                        learn_p=None,
                        div_in=2,
                    ),
                    swap=nonloc_scope.swap,
                )
            else:
                ks = list(map(int, kernels.split('-')))

                if init_layer is None:
                    _block = layers.ChannelCouplingBlock
                    _mask_type = 'channel'
                    div_in = 2
                    mult_out = 1
                else:
                    _block = layers.MaskedCouplingBlock
                    _mask_type = 'checkerboard'
                    div_in = 1
                    mult_out = 2

                nonloc_scope.swap = not nonloc_scope.swap
                _mask_type += '1' if nonloc_scope.swap else '0'

                nnet = []
                if not first_resblock and preact:
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(_weight_layer(fc)(
                    initial_size[0] // div_in, idim, ks[0], 1, ks[0] // 2))
                if batchnorm:
                    nnet.append(layers.MovingBatchNorm2d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(_weight_layer(fc)(idim, idim, k, 1, k // 2))
                    if batchnorm:
                        nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout:
                    nnet.append(nn.Dropout2d(dropout, inplace=True))
                nnet.append(_weight_layer(fc)(
                    idim, initial_size[0] * mult_out, ks[-1], 1, ks[-1] // 2))
                if batchnorm:
                    nnet.append(layers.MovingBatchNorm2d(initial_size[0]))

                return _block(initial_size[0], nn.Sequential(*nnet), mask_type=_mask_type)

        if init_layer is not None:
            chain.append(init_layer)
        if first_resblock and actnorm:
            chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm:
            chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, h, w = initial_size
            for i in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc,
                                       first_resblock=first_resblock and (i == 0)))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic:
                    chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc))
                if fc_actnorm:
                    chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm:
                        chain.append(_actnorm(initial_size, True))

        super(StackedCouplingBlocks, self).__init__(chain)
