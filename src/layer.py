import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import constant_
import torch.nn.functional as F
from itertools import repeat
import collections.abc


def _ntuple(n):
    """Translate a float value to n-dim tuple
    Examples
    --------
    >>> a = _ntuple(2)
    >>> a(10)
    (10, 10)
    >>> b = _ntuple(5)
    >>> b(-2)
    (-2, -2, -2, -2, -2)
    Aliases
    -------
    >>> _single(2)
    (2,)
    >>> _pair(2)
    (2, 2)
    >>> _triple(2)
    (2, 2, 2)
    >>> _quadruple(2)
    (2, 2, 2, 2)
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _generate_hold_kernel(in_channels, zoh_kernel_size, order):
    """Convolve a zero-order hold kernel with the size of *zoh_kernel_size* *order* times
    Examples
    --------
    >>> _generate_hold_kernel(1, 2, 0)
    tensor([[[[1., 1.],
              [1., 1.]]]])
    >>> _generate_hold_kernel(1, 2, 1)
    tensor([[[[1., 2., 1.],
              [2., 4., 2.],
              [1., 2., 1.]]]])
    """
    zoh_kernel_size = _pair(zoh_kernel_size)

    # Zero-order hold kernel
    zoh_kernel = torch.Tensor(1, 1, *zoh_kernel_size)
    constant_(zoh_kernel, 1.0)
    tmp_kernel = zoh_kernel.clone()
    for i in range(order):
        tmp_kernel = F.conv2d(
            tmp_kernel, zoh_kernel, bias=None, stride=(1, 1),
            padding=((zoh_kernel_size[1]+1)//2,
                     (zoh_kernel_size[0]+1)//2),
            dilation=(1, 1), groups=1)
    return torch.repeat_interleave(tmp_kernel, in_channels, dim=0)


class DownSampling2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 order=0, hold_mode='hold_first', bias_mode='bias_first'):
        super(DownSampling2d, self).__init__()
        self.hold_mode = hold_mode
        self.bias_mode = bias_mode

        if hold_mode == "hold_first":
            hold_in_channels = in_channels
        elif hold_mode == "hold_last":
            hold_in_channels = out_channels
        elif hold_mode == "kernel_conv":
            hold_in_channels = None
        else:
            raise(NotImplementedError())

        if hold_mode == "kernel_conv":
            self.conv = KernelConv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode, order=order)
        elif bias_mode == "bias_first":
            self.hold_conv = HoldConv2d(
                in_channels=hold_in_channels, zoh_kernel_size=stride,
                order=order, stride=1, padding=0, bias=bias,
                padding_mode='zeros')
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=False, padding_mode=padding_mode)
        elif bias_mode == "bias_last":
            self.hold_conv = HoldConv2d(
                in_channels=hold_in_channels, zoh_kernel_size=stride,
                order=order, stride=1, padding=0, bias=False,
                padding_mode='zeros')
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        if self.hold_mode == "hold_first":
            x = self.hold_conv(x)
            x = self.conv(x)
        elif self.hold_mode == "hold_last":
            raise(NotImplementedError())
        else:
            x = self.conv(x)

        return x


class HoldConv2d(nn.Module):
    def __init__(self, in_channels, zoh_kernel_size, order=0, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros'):

        super(HoldConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.zoh_kernel_size = _pair(zoh_kernel_size)
        self.order = order
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = in_channels
        self.padding_mode = padding_mode

        kernel = _generate_hold_kernel(in_channels,
                                       self.zoh_kernel_size,
                                       self.order)
        self.kernel = Parameter(kernel, requires_grad=False)
        self.kernel_size = self.kernel.size()[2:]

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        expanded_padding = (self.kernel_size[1] // 2,
                            (self.kernel_size[1]-1) // 2,
                            self.kernel_size[0] // 2,
                            (self.kernel_size[0]-1) // 2)
        input = F.pad(input, expanded_padding)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.kernel, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.kernel, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class KernelConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', order=0):
        super(KernelConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

        self.order = order
        kernel = _generate_hold_kernel(in_channels,
                                       self.stride,
                                       self.order)
        self.hold_kernel = Parameter(kernel, requires_grad=False)
        self.hold_kernel_size = self.hold_kernel.size()[2:]

    def forward(self, input):
        padding = ((self.hold_kernel_size[1]+1)//2,
                   (self.hold_kernel_size[0]+1)//2)
        kernel = F.conv2d(
                    self.weight, self.hold_kernel, bias=None, stride=(1, 1),
                    padding=padding,
                    dilation=(1, 1), groups=self.in_channels)
        kernel_size = kernel.size()[2:]
        expanded_padding = (kernel_size[1] // 2 - padding[1],
                            (kernel_size[1]-1) // 2 - padding[1],
                            kernel_size[0] // 2 - padding[0],
                            (kernel_size[0]-1) // 2 - padding[0])
        input = F.pad(input, expanded_padding)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            kernel, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, kernel, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class UpSampling2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 order=0, hold_mode='hold_first', bias_mode='bias_first'):
        super(UpSampling2d, self).__init__()
        self.hold_mode = hold_mode
        self.bias_mode = bias_mode

        if hold_mode == "hold_first":
            hold_in_channels = in_channels
        elif hold_mode == "hold_last":
            hold_in_channels = out_channels
        elif hold_mode == "kernel_conv":
            raise(NotImplementedError())
        else:
            raise(NotImplementedError())

        if hold_mode == "kernel_conv":
            pass
        elif bias_mode == "bias_first":
            self.hold_conv = HoldConv2d(
                in_channels=hold_in_channels, zoh_kernel_size=stride,
                order=order, stride=1, padding=0, bias=False,
                padding_mode='zeros')
            self.trans_conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, padding_mode=padding_mode)
        elif bias_mode == "bias_last":
            self.hold_conv = HoldConv2d(
                in_channels=hold_in_channels, zoh_kernel_size=stride,
                order=order, stride=1, padding=0, bias=bias,
                padding_mode='zeros')
            self.trans_conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=False, padding_mode=padding_mode)

    def forward(self, x):
        if self.hold_mode == "hold_first":
            raise(NotImplementedError())
        elif self.hold_mode == "hold_last":
            x = self.trans_conv(x)
            x = self.hold_conv(x)
        else:
            x = self.trans_conv(x)

        return x
