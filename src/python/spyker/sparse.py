from spyker.utils import *
import spyker.spyker_plugin as impl


def code(input, time, sort=True):
    input_ = to4(wrap(input))
    return impl.sparse.code(input_, time, sort)


def conv(input, kernel, threshold, stride=1, pad=0):
    stride, pad = expand2(stride), expand4(pad)
    return impl.sparse.conv(input, kernel, threshold, stride, pad)


def gather(input, dtype='u8'):
    return impl.sparse.gather(input, dtype)


def pool(input, kernel, stride=None, pad=0):
    if stride is None:
        stride = kernel
    kernel = expand2(kernel)
    stride = expand2(stride)
    pad = expand4(pad)
    return impl.sparse.pool(input, kernel, stride, pad)


def inhibit(input):
    return impl.sparse.inhibit(input)


def convwta(input, radius, count):
    radius = expand2(radius)
    return impl.sparse.convwta(input, radius, count)
