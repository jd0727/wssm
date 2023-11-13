import collections
from collections.abc import Iterable
from itertools import repeat

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# <editor-fold desc='激活函数'>
# Swish
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(1.0))

    def forward(self, x):
        x = x * torch.sigmoid(x * self.beta)
        return x


# SiLU
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


# Mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


# Relu6
class ReLU6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, min=0, max=6)
        return x


class HSiLU(nn.Module):
    def forward(self, x):
        out = x * (torch.clamp(x, min=-3, max=3) / 6 + 0.5)
        return out


class HSigmoid(nn.Module):
    def forward(self, x):
        out = torch.clamp(x, min=-3, max=3) / 6 + 0.5
        return out


class ACT:
    LK = 'lk'
    RELU = 'relu'
    SIG = 'sig'
    RELU6 = 'relu6'
    MISH = 'mish'
    SILU = 'silu'
    HSILU = 'hsilu'
    HSIG = 'hsig'
    SWISH = 'swish'
    TANH = 'tanh'
    NONE = None

    @staticmethod
    def build(act_name=None):
        if isinstance(act_name, nn.Module):
            act = act_name
        elif act_name is None or act_name == '':
            act = None
        elif act_name == ACT.LK:
            act = nn.LeakyReLU(0.1, inplace=True)
        elif act_name == ACT.RELU:
            act = nn.ReLU(inplace=True)
        elif act_name == ACT.SIG:
            act = nn.Sigmoid()
        elif act_name == ACT.SWISH:
            act = Swish()
        elif act_name == ACT.RELU6:
            act = ReLU6()
        elif act_name == ACT.MISH:
            act = Mish()
        elif act_name == ACT.SILU:
            act = SiLU()
        elif act_name == ACT.HSILU:
            act = HSiLU()
        elif act_name == ACT.HSIG:
            act = HSigmoid()
        elif act_name == ACT.TANH:
            act = nn.Tanh()
        else:
            raise Exception('err act name' + str(act_name))
        return act


# </editor-fold>

# <editor-fold desc='卷积子模块'>
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _auto_pad(kernel_size, dilation):
    if isinstance(kernel_size, tuple):
        kernel_size = _pair(kernel_size)
        dilation = _pair(dilation)
        padding = ((kernel_size[0] - 1) * dilation[0] // 2, (kernel_size[1] - 1) * dilation[1] // 2)
        return padding
    else:
        return (kernel_size - 1) * dilation // 2


# Conv
class C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(C, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
            bias=bias, groups=groups, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups, bias=self.conv.bias is not None)

    @staticmethod
    def convert(c):
        if isinstance(c, C):
            return c
        elif isinstance(c, RCpa):
            ct = C(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            return ct
        elif isinstance(c, DC):
            ct = C(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            return ct
        else:
            raise Exception('err module ' + c.__class__.__name__)


class Cpa(C):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(Cpa, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), bias=bias, **kwargs)


class Ck1(C):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=1, bias=True, **kwargs):
        super(Ck1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
            dilation=dilation, groups=groups, bias=bias, **kwargs)


class Ck1s1(C):
    def __init__(self, in_channels, out_channels, groups=1, dilation=1, bias=True, **kwargs):
        super(Ck1s1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
            groups=groups, bias=bias, **kwargs)


class Ck3(C):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=1, bias=True, **kwargs):
        super(Ck3, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, groups=groups, bias=bias, **kwargs)


class Ck3s1(C):
    def __init__(self, in_channels, out_channels, groups=1, dilation=1, bias=True, **kwargs):
        super(Ck3s1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=dilation,
            dilation=dilation, groups=groups, bias=bias, **kwargs)


# Conv+Act
class CA(C):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, act=None, **kwargs):
        super(CA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.act = ACT.build(act) if act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x) if self.act else x
        return x

    @staticmethod
    def convert(c):
        if isinstance(c, CA):
            return c
        elif isinstance(c, RCpaA):
            ct = CA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.act = c.act
            return ct
        elif isinstance(c, DCA):
            ct = CA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.act = c.act
            return ct
        else:
            raise Exception('err module ' + c.__class__.__name__)


# Conv+BN+Act padding=auto
class CpaA(CA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=True, act=None, **kwargs):
        super(CpaA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), bias=bias, act=act, **kwargs)


# Conv+BN+Act padding=auto
class Ck3A(CA):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, bias=True, act=None, **kwargs):
        super(Ck3A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
            dilation=dilation, groups=groups, padding=dilation, bias=bias, act=act, **kwargs)


class Ck3s1A(CA):
    def __init__(self, in_channels, out_channels, dilation=1, groups=1, bias=True, act=None, **kwargs):
        super(Ck3s1A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
            dilation=dilation, groups=groups, padding=dilation, bias=bias, act=act, **kwargs)


class Ck1A(CA):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, bias=True, act=None, **kwargs):
        super(Ck1A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
            dilation=1, groups=groups, padding=0, bias=bias, act=act, **kwargs)


class Ck1s1A(CA):
    def __init__(self, in_channels, out_channels, groups=1, bias=True, act=None, **kwargs):
        super(Ck1s1A, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
            dilation=1, groups=groups, padding=0, bias=bias, act=act, **kwargs)


class CB(C):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn=True, **kwargs):
        super(CB, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not bn, groups=groups, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        return x


class Ck1s1B(CB):
    def __init__(self, in_channels, out_channels, groups=1, dilation=1, bn=True, **kwargs):
        super(Ck1s1B, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation,
            groups=groups, bn=bn, **kwargs)


# Conv+Bn+Act
class CBA(CA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(CBA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not bn, groups=groups, act=act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x

    @staticmethod
    def convert(c):
        if isinstance(c, CBA):
            return c
        elif isinstance(c, RCpaBA):
            ct = CBA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.bn = c.bn
            ct.act = c.act
            return ct
        elif isinstance(c, DCBA):
            ct = CBA(**c.config)
            conv_eq = c.conv_eq
            ct.conv.weight = conv_eq.weight
            ct.conv.bias = conv_eq.bias
            ct.bn = c.bn
            ct.act = c.act
            return ct
        else:
            raise Exception('err module ' + c.__class__.__name__)


# Conv+Bn+Act padding=0
class Cp0BA(CBA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(CBA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0,
            dilation=dilation, groups=groups, bn=bn, act=act, **kwargs)


# Conv+BN+Act padding=auto
class CpaBA(CBA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(CpaBA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=_auto_pad(kernel_size, dilation), bn=bn, act=act)


class CpadwBA(CpaBA):
    def __init__(self, channels, kernel_size=1, stride=1, dilation=1, bn=True, act=None, **kwargs):
        super(CpadwBA, self).__init__(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=channels, bn=bn, act=act, **kwargs)


class Ck1BA(CBA):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=1, bn=True, act=None, **kwargs):
        super(Ck1BA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0,
            dilation=dilation, groups=groups, bn=bn, act=act, **kwargs)


class Ck1s1BA(Ck1BA):
    def __init__(self, in_channels, out_channels, groups=1, bn=True, act=None, **kwargs):
        super(Ck1s1BA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, stride=1, groups=groups,
            bn=bn, act=act, **kwargs)


class Ck3BA(CBA):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=1, bn=True, act=None, **kwargs):
        super(Ck3BA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, groups=groups, bn=bn, act=act, **kwargs)


class Ck3s1BA(Ck3BA):
    def __init__(self, in_channels, out_channels, groups=1, dilation=1, bn=True, act=None, **kwargs):
        super(Ck3s1BA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, stride=1, groups=groups, dilation=dilation,
            bn=bn, act=act, **kwargs)


# ConvTrans
class CT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=0, bias=True, **kwargs):
        super(CT, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=_pair(kernel_size),
                                       stride=_pair(stride), groups=groups, padding=_pair(padding),
                                       output_padding=_pair(output_padding),
                                       dilation=dilation, bias=bias, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    output_padding=self.conv.output_padding, kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups, bias=self.conv.bias is not None)


# ConvTrans+Act
class CTpa(CT):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=True, **kwargs):
        padding = (kernel_size - 1) * dilation // 2
        output_padding = (2 * padding - kernel_size) % stride
        super(CTpa, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=groups,
                                   output_padding=output_padding, bias=bias, **kwargs)


# ConvTrans+Act
class CTA(CT):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=0, bias=True, act=None, **kwargs):
        super(CTA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups,
                                  output_padding=output_padding, bias=bias, **kwargs)
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x) if self.act else x
        return x


# ConvTrans+BN+Act
class CTpaA(CTA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=True, act=None, **kwargs):
        padding = (kernel_size - 1) * dilation // 2
        output_padding = (2 * padding - kernel_size) % stride
        super(CTpaA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, groups=groups,
                                    output_padding=output_padding, bias=bias, act=act, **kwargs)


# ConvTrans+BN+Act
class CTBA(CTA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=0, bn=True, act=None, **kwargs):
        super(CTBA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=groups,
                                   output_padding=output_padding, bias=not bn, act=act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


# ConvTrans+BN+Act padding=auto
class CTpaBA(CTBA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        padding = (kernel_size - 1) * dilation // 2
        output_padding = (2 * padding - kernel_size) % stride
        super(CTpaBA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=padding, output_padding=output_padding, bn=bn, act=act, **kwargs)


# ConvTrans+BN+Act
class CTk3BA(CTBA):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        padding = dilation
        output_padding = (2 * padding - 3) % stride
        super(CTk3BA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
            dilation=dilation, groups=groups, padding=padding, output_padding=output_padding, bn=bn, act=act, **kwargs)


# </editor-fold>

# <editor-fold desc='Rep卷积子模块'>
class RCpa(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(RCpa, self).__init__()

        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
            bias=bias, groups=groups, **kwargs)

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
            stride=_pair(stride), padding=(0, 0), dilation=(1, 1),
            bias=bias, groups=groups, **kwargs)

        self.has_shortcut = in_channels == out_channels and stride == 1
        self._conv_eq = None

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups, bias=self.conv.bias is not None)

    @property
    def conv_eq(self):
        if self._conv_eq is None:
            self._conv_eq = self._get_conv_eq()
        return self._conv_eq

    @property
    def conv_wb(self):
        return self.conv.weight, self.conv.bias

    @conv_wb.setter
    def conv_wb(self, conv_wb):
        self.conv.weight, self.conv.bias = conv_wb

    @property
    def conv_1_wb(self):
        return self.conv_1.weight, self.conv_1.bias

    @property
    def shortcut_wb(self):
        weight = torch.eye(self.conv_1.weight.size(1))[:, :, None, None]
        weight = weight.repeat(self.conv_1.groups, 1, 1, 1)
        return weight, None

    @property
    def conv_eq_wb(self):
        weight, bias = self.conv_wb
        weight_1, bias_1 = self.conv_1_wb
        kernel_size = self.conv.kernel_size
        pad_pre = (kernel_size[0] // 2, kernel_size[1] // 2)
        pad = (pad_pre[0], kernel_size[0] - pad_pre[0] - 1, pad_pre[1], kernel_size[1] - pad_pre[1] - 1)
        weight_eq = weight + F.pad(weight_1, pad=pad)
        bias_eq = None if bias is None else bias + bias_1
        if self.has_shortcut:
            weight_sc, bias_sc = self.shortcut_wb
            weight_eq = weight_eq + F.pad(weight_sc, pad=pad)
            bias_eq = bias_eq if bias is None or bias_sc is None else bias_eq + bias_sc
        return weight_eq, bias_eq

    # @conv_eq_wb.setter
    # def conv_eq_wb(self, conv_eq_wb):
    #     weight_eq, bias_eq = conv_eq_wb

    def _get_conv_eq(self):
        weight_eq, bias_eq = self.conv_eq_wb
        bias = bias_eq is not None
        _conv_eq = nn.Conv2d(in_channels=self.conv.in_channels,
                             out_channels=self.conv.out_channels,
                             kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                             padding=self.conv.padding, dilation=self.conv.dilation,
                             groups=self.conv.groups, bias=bias)
        _conv_eq.weight.data = weight_eq
        if bias:
            _conv_eq.bias.data = bias_eq
        return _conv_eq

    def train(self, mode: bool = True):
        last_mode = self.training
        super(RCpa, self).train(mode)
        if last_mode and not mode:
            self._conv_eq = self._get_conv_eq()

    def forward(self, x):
        if self.training:
            out = self.conv(x) + self.conv_1(x)
            out = out + x if self.has_shortcut else out
        else:
            out = self.conv_eq(x)
        return out

    @staticmethod
    def convert(rc):
        if isinstance(rc, RCpa):
            return rc
        elif isinstance(rc, C):  # 不确保相等
            config = rc.config
            del config['padding']
            rct = RCpa(**config)
            rct.conv.weight = rc.conv.weight
            rct.conv.bias = rc.conv.bias
            return rct
        elif isinstance(rc, DC):
            # 没写
            return RCpa.convert(C.convert(rc))
        else:
            raise Exception('err module ' + rc.__class__.__name__)


class RCk3(RCpa):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(RCk3, self).__init__(in_channels, out_channels, kernel_size=3, stride=stride,
                                   dilation=dilation, groups=groups, bias=bias, **kwargs)


class RCk3s1(RCk3):
    def __init__(self, in_channels, out_channels, dilation=1, groups=1, bias=True, **kwargs):
        super(RCk3s1, self).__init__(in_channels, out_channels, stride=1,
                                     dilation=dilation, groups=groups, bias=bias, **kwargs)


class RCpaA(RCpa):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, act=None, **kwargs):
        super(RCpaA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.act = ACT.build(act) if act else None

    def forward(self, x):
        out = super(RCpaA, self).forward(x)
        return self.act(out) if self.act else out


class RCk3A(RCpaA):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(RCk3A, self).__init__(in_channels, out_channels, kernel_size=3, stride=stride,
                                    dilation=dilation, groups=groups, bn=bn, act=act, **kwargs)


class RCk3s1A(RCk3A):
    def __init__(self, in_channels, out_channels, dilation=1, groups=1, bn=True, act=None, **kwargs):
        super(RCk3s1A, self).__init__(in_channels, out_channels, stride=1,
                                      dilation=dilation, groups=groups, bn=bn, act=act, **kwargs)


class RCpaBA(RCpaA):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(RCpaBA, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     dilation=dilation, groups=groups, bias=not bn, act=act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.bn_1 = nn.BatchNorm2d(out_channels) if bn else None
        self.bn_sc = nn.BatchNorm2d(out_channels) if bn else None

    def _fuse_bn(self, weight, bias, bn):
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        bias_fusd = beta - running_mean * gamma / std
        bias_fusd = bias_fusd if bias is None else bias_fusd + bias * gamma / std
        weight_fusd = weight * t
        return weight_fusd, bias_fusd

    @property
    def conv_wb(self):
        return self._fuse_bn(self.conv.weight, self.conv.bias, self.bn)

    @property
    def conv_1_wb(self):
        return self._fuse_bn(self.conv_1.weight, self.conv_1.bias, self.bn_1)

    @property
    def shortcut_wb(self):
        weight = torch.eye(self.conv_1.weight.size(1))[:, :, None, None]
        weight = weight.repeat(self.conv_1.groups, 1, 1, 1)
        return self._fuse_bn(weight, None, self.bn_sc)

    def forward(self, x):
        if self.training:
            out = self.bn(self.conv(x)) + self.bn_1(self.conv_1(x))
            out = out + self.bn_sc(x) if self.has_shortcut else out
        else:
            out = self.conv_eq(x)
        return self.act(out) if self.act else out

    @property
    def config(self):
        return dict(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                    kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv.groups, bn=self.bn is not None)


class RCk3BA(RCpaBA):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(RCk3BA, self).__init__(
            in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, groups=groups, bn=bn, act=act,
            **kwargs)


class RCk3s1BA(RCk3BA):
    def __init__(self, in_channels, out_channels, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(RCk3s1BA, self).__init__(
            in_channels, out_channels, stride=1, dilation=dilation, groups=groups, bn=bn, act=act, **kwargs)


# if __name__ == '__main__':
#     layer = RCpaB(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation=1, groups=3, )
#     print(layer.training)
#     layer.eval()
#     x = torch.rand(size=(1, 6, 5, 5), dtype=torch.float32)
#     y1 = layer(x)
#     y2 = layer.conv_eq(x)
#     print(y1 - y2)


# </editor-fold>

# <editor-fold desc='Dw卷积子模块'>
class DC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(DC, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
            stride=(1, 1), padding=(0, 0), dilation=(1, 1),
            bias=False, groups=groups, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=_pair(kernel_size),
            stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation),
            bias=bias, groups=1, **kwargs)

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        return x

    @property
    def conv_eq(self):
        bias = self.conv.bias is not None
        _conv_eq = nn.Conv2d(in_channels=self.conv.in_channels,
                             out_channels=self.conv.out_channels,
                             kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                             padding=self.conv.padding, dilation=self.conv.dilation,
                             groups=self.conv.groups, bias=bias)
        _conv_eq.weight.data = self.conv.weight.data * self.conv_1.weight.data
        if bias:
            _conv_eq.bias.data = self.conv.bias.data
        return _conv_eq

    @property
    def config(self):
        return dict(in_channels=self.conv_1.in_channels, out_channels=self.conv_1.out_channels,
                    kernel_size=self.conv.kernel_size,
                    stride=self.conv.stride, padding=self.conv.padding,
                    dilation=self.conv.dilation, groups=self.conv_1.groups, bias=self.conv.bias is not None)


class DCA(DC):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, act=None, **kwargs):
        super(DCA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.act = ACT.build(act) if act else None

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        x = self.act(x) if self.act else x
        return x


class DCB(DC):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn=True, **kwargs):
        super(DCB, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not bn, groups=groups, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        x = self.bn(x) if self.bn else x
        return x


class DCBA(DCA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(DCBA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=not bn, groups=groups, act=act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(self.conv_1(x))
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


class DCpa(DC):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=True, **kwargs):
        super(DCpa, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=(kernel_size - 1) * dilation // 2, bias=bias, **kwargs)


class DCpaBA(DCBA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(DCpaBA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=(kernel_size - 1) * dilation // 2, bn=bn, act=act)


class DCpaA(DCA):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bias=True, act=None, **kwargs):
        super(DCpaA, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=(kernel_size - 1) * dilation // 2, act=act, bias=bias)


# </editor-fold>

# <editor-fold desc='检测框层'>
# 基于基础尺寸和长宽比生成anchor_sizes
def generate_anchor_sizes(base, scales=(8, 16, 32), wh_ratios=(0.5, 1, 2)):
    wh_ratios = torch.sqrt(torch.Tensor(wh_ratios))[None, :]
    scales = torch.Tensor(scales)[:, None]
    ws = scales * wh_ratios * base
    hs = scales / wh_ratios * base
    anchor_sizes = torch.stack([ws, hs], dim=-1)
    anchor_sizes = anchor_sizes.reshape(-1, 2)
    return anchor_sizes


# 有先验框
class AnchorLayer(nn.Module):
    def __init__(self, anchor_sizes, stride, feat_size=(0, 0)):
        super(AnchorLayer, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.stride = stride
        self.feat_size = feat_size

    @property
    def anchor_sizes(self):
        return self._anchor_sizes

    @anchor_sizes.setter
    def anchor_sizes(self, anchor_sizes):
        anchor_sizes = torch.Tensor(anchor_sizes)
        if len(anchor_sizes.size()) == 1:
            anchor_sizes = torch.unsqueeze(anchor_sizes, dim=0)
        self._anchor_sizes = anchor_sizes
        self.Na = anchor_sizes.size(0)

    @staticmethod
    def generate_offset(Wf, Hf, anchor_sizes):
        Na = anchor_sizes.size(0)  # (Na,2)
        x = torch.arange(Wf)[None, :].expand(Hf, Wf)
        y = torch.arange(Hf)[:, None].expand(Hf, Wf)
        xy_offset = torch.stack([x, y], dim=2)  # (Hf,Wf,2)
        if Na == 1:
            wh_offset = anchor_sizes[None, :, :].expand(Hf, Wf, 2).contiguous()  # (Hf,Wf,2)
            return xy_offset, wh_offset
        else:
            xy_offset = xy_offset.unsqueeze(dim=2).expand(Hf, Wf, Na, 2).contiguous()  # (Hf, Wf, Na, 2)
            wh_offset = anchor_sizes[None, None, :, :].expand(Hf, Wf, Na, 2).contiguous()  # (Hf, Wf, Na, 2)
            return xy_offset, wh_offset

    @staticmethod
    def generate_anchor_sizes(xy_offset, wh_offset, stride):
        anchors = torch.cat([(xy_offset + 0.5) * stride, wh_offset], dim=-1)
        xc, yc, w, h = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
        anchors[..., 0], anchors[..., 2] = xc - w / 2, xc + w / 2
        anchors[..., 1], anchors[..., 3] = yc - h / 2, yc + h / 2
        anchors = anchors.reshape(-1, 4)
        return anchors

    @property
    def feat_size(self):
        return (self.Wf, self.Hf)

    @feat_size.setter
    def feat_size(self, feat_size):
        (self.Wf, self.Hf) = feat_size if isinstance(feat_size, Iterable) else (feat_size, feat_size)
        self.xy_offset, self.wh_offset = AnchorLayer.generate_offset(self.Wf, self.Hf, self.anchor_sizes)
        self.anchors = AnchorLayer.generate_anchor_sizes(self.xy_offset, self.wh_offset, self.stride)
        self.num_anchor = self.Na * self.Wf * self.Hf


class AnchorLayerImg(AnchorLayer):
    def __init__(self, anchor_sizes, stride, img_size=(0, 0)):
        super().__init__(anchor_sizes, stride, AnchorLayerImg.calc_feat_size(img_size, stride))

    @staticmethod
    def calc_feat_size(img_size, stride):
        (W, H) = img_size if isinstance(img_size, Iterable) else (img_size, img_size)
        return (int(math.ceil(W / stride)), int(math.ceil(H / stride)))

    @property
    def img_size(self):
        return (self.Wf * self.stride, self.Hf * self.stride)

    @img_size.setter
    def img_size(self, img_size):
        self.feat_size = AnchorLayerImg.calc_feat_size(img_size, self.stride)


class AnchorSampLayer(AnchorLayer):
    def __init__(self, anchor_sizes, stride, feat_size=(0, 0), feat_size_samp=(0, 0)):
        super(AnchorSampLayer, self).__init__(anchor_sizes=anchor_sizes, stride=stride, feat_size=feat_size)
        self.feat_size_samp = feat_size_samp

    @property
    def feat_size_samp(self):
        return (self.Wfs, self.Hfs)

    @feat_size_samp.setter
    def feat_size_samp(self, feat_size_samp):
        (self.Wfs, self.Hfs) = feat_size_samp if isinstance(feat_size_samp, Iterable) else \
            (feat_size_samp, feat_size_samp)
        self.xys_offset, self.whs_offset = AnchorLayer.generate_offset(self.Wfs, self.Hfs, self.anchor_sizes)


# </editor-fold>

# <editor-fold desc='初始化'>

def init_sig(bias, prior_prob=0.1):
    nn.init.constant_(bias.data, -math.log((1 - prior_prob) / prior_prob))
    return bias


def init_xavier(weight):
    """Caffe2 XavierFill Implementation"""
    fan_in = weight.numel() / weight.size(0)
    scale = math.sqrt(3 / fan_in)
    nn.init.uniform_(weight, -scale, scale)
    return weight


def init_msra(weight):
    """Caffe2 MSRAFill Implementation"""
    fan_in = weight.numel() / weight.size(0)
    scale = math.sqrt(2 / fan_in)
    nn.init.uniform_(weight, -scale, scale)
    return weight


# </editor-fold>

# <editor-fold desc='自定义loss'>

# def binaryfocal_loss(pred, target, alpha=0.5, gamma=2, reduction='sum'):
#     loss_ce = -target * torch.log(pred + 1e-8) - (1 - target) * torch.log(1 - pred + 1e-8)
#     prop = target * pred + (1 - target) * (1 - pred)
#     alpha_full = target * alpha + (1 - target) * (1 - alpha)
#     loss = loss_ce * alpha_full * (1 - prop) ** gamma
#     if reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()
#     elif reduction == 'none':
#         return loss
#     else:
#         raise Exception('err reduction')

def _reduct_loss(loss, reduction='sum'):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise Exception('err reduction')


def binary_focal_loss_with_logits(pred, target, alpha=0.5, gamma=2, reduction='sum'):
    pred_prob = torch.sigmoid(pred)  # prob from logits
    pt = target * pred_prob + (1 - target) * (1 - pred_prob)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = (1 - pt) ** gamma
    focal_weight = alpha_factor * modulating_factor
    loss = F.binary_cross_entropy_with_logits(input=pred, target=target, weight=focal_weight, reduction=reduction)
    return loss


def binary_focal_loss(pred, target, alpha=0.5, gamma=2, reduction='sum'):
    pt = target * pred + (1 - target) * (1 - pred)
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = (1 - pt) ** gamma
    focal_weight = alpha_factor * modulating_factor
    loss = F.binary_cross_entropy(pred, target, weight=focal_weight, reduction=reduction)
    return loss


def binary_qfocal_loss(pred, target, alpha=0.5, gamma=2, reduction='sum'):
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.abs(pred - target) ** gamma
    focal_weight = alpha_factor * modulating_factor
    loss = F.binary_cross_entropy(pred, target, weight=focal_weight, reduction=reduction)
    return loss


# 默认最后一个维度是分类，pred和target同大小
def focal_loss(pred, target, alphas=(1, 2, 3), gamma=2, reduction='sum'):
    pred_sft = torch.softmax(pred, dim=-1)
    alphas = torch.Tensor(alphas).to(pred.device)
    focal_weight = alphas * (1 - pred_sft).pow(gamma)
    loss = -torch.sum(torch.log(pred_sft + 1e-7) * target * focal_weight, dim=-1)
    return _reduct_loss(loss, reduction)


# </editor-fold>


# <editor-fold desc='模型替换'>

def model_apply(model, func_dct):
    for name, sub_module in model.named_children():
        if sub_module.__class__ in func_dct.keys():
            func = func_dct[sub_module.__class__]
            if isinstance(func, nn.Module):
                sub_module_replalce = func
            else:
                sub_module_replalce = func(sub_module)
            setattr(model, name, sub_module_replalce)
        else:
            model_apply(sub_module, func_dct)
    return model


def model_react(model, act_old, act_new):
    func_dct = {ACT.build(act_old).__class__: lambda x: ACT.build(act_new)}
    model_apply(model, func_dct)
    return model


def model_rc2c(model):
    func_dct = {RCpa: C.convert, RCpaA: CA.convert, RCpaBA: CBA.convert, }
    model_apply(model, func_dct)
    return model


def model_dc2c(model):
    func_dct = {DC: C.convert, DCA: CA.convert, DCBA: CBA.convert, }
    model_apply(model, func_dct)
    return model

# def replace_conv2convdwseq(model):
#     def generete_convdwseq(conv):
#         assert isinstance(conv, CBA), 'conv err'
#         core = conv.conv
#         if core.kernel_size[0] == 1 or core.groups == core.out_channels:
#             return conv
#         seq = nn.Sequential(
#             CBA(in_channels=core.in_channels, out_channels=core.in_channels, kernel_size=core.kernel_size[0],
#                 stride=core.stride[0], padding=int(core.padding[0]), dilation=core.dilation[0],
#                 groups=core.in_channels, bn=not conv.bn is None, act=conv.act),
#             CBA(in_channels=core.in_channels, out_channels=core.out_channels, kernel_size=1,
#                 stride=1, padding=0, dilation=1, groups=1, bn=not conv.bn is None, act=conv.act)
#         )
#         return seq
#
#     def _replace_conv2convdwseq(module):
#         for name, sub_module in module.named_children():
#             if isinstance(sub_module, CBA):
#                 sub_module = generete_convdwseq(sub_module)
#                 setattr(module, name, sub_module)
#             else:
#                 _replace_conv2convdwseq(sub_module)
#         return None
#
#     if isinstance(model, CBA):
#         return generete_convdwseq(model)
#     model = copy.deepcopy(model)
#     _replace_conv2convdwseq(model)
#     return model

# </editor-fold>


# class TM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = DCpaA(4, 5, kernel_size=3, act=ACT.RELU)
#         self.c2 = DCpaA(5, 6, kernel_size=3, act=ACT.LK)
#
#     def forward(self, x):
#         return self.c2(self.c1(x))


# if __name__ == '__main__':
#     model = TM()
#     x = torch.rand(1, 4, 3, 3)
#     print(model)
#     y1 = model(x)
#     model_dc2c(model)
#     # model_react(model, ACT.RELU, ACT.SWISH)
#     print(model)
#     y2 = model(x)
#     print(y2 - y1)
