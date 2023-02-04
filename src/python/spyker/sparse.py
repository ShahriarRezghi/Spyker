from spyker.utils import *
import spyker.spyker_plugin as impl


def code(array, time, sort=True):
    input_ = to4(wrap(array))
    return impl.sparse.code(input_, time, sort)


def conv(array, kernel, threshold, stride=1, pad=0):
    stride, pad = expand2(stride), expand4(pad)
    return impl.sparse.conv(array, kernel, threshold, stride, pad)


def gather(array, dtype='u8'):
    return impl.sparse.gather(array, dtype)


def pool(array, kernel, stride=None, pad=0):
    if stride is None:
        stride = kernel
    kernel = expand2(kernel)
    stride = expand2(stride)
    pad = expand4(pad)
    return impl.sparse.pool(array, kernel, stride, pad)


def inhibit(array):
    return impl.sparse.inhibit(array)


def convwta(array, radius, count):
    radius = expand2(radius)
    return impl.sparse.convwta(array, radius, count)
