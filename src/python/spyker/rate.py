from spyker.utils import *
import spyker.spyker_plugin as impl


def code(input, time, sort=True, dtype=impl.u8):
    """
    Apply rank coding

    Parameters
    ----------
    input : spyker.Tensor or torch.tensor or numpy.ndarray
        Input tensor
    time : int
        Number of time steps
    sort : bool, optional
        Whether to sort the values or not (default True). Sorting might increase accuracy but it deceases performance

    Returns
    -------
    spyker.Tensor or torch.tensor or numpy.ndarray
        Rank coding output tensor
    """

    input_ = least2(wrap(input))
    shape = impl.shape.code(input_.shape, time)
    output = create(input, dtype, shape)
    impl.rate.code(input_, wrap(output), time, sort)
    return output


def fire(input, threshold=.0, dtype=impl.u8):
    """
    Apply integrate-and-fire mechanism

    If the input is already thresholded then there is no need to pass threshold to this function.

    Parameters
    ----------
    input : spyker.Tensor or torch.tensor or numpy.ndarray
        Input tensor
    threshold : float, optional
        Threshold of firing (default 0)

    Returns
    -------
    spyker.Tensor or torch.tensor or numpy.ndarray
        Integrate-and-fire output tensor
    """

    input_ = wrap(input)
    output = create(input, dtype, input_.shape)
    impl.rate.fire(input_, wrap(output), threshold)
    return output


def gather(input, threshold=.0, dtype=impl.u8):
    """
    Gather temporal information

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    input : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Input tensor
    threshold : float, optional
        Threshold of firing (default 0.0). Only used when input has type F32

    Returns
    -------
    spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Gathered output tensor
    """

    input_ = least3(wrap(input))
    output = create(input, dtype, impl.shape.gather(input_.shape))
    impl.rate.gather(input_, wrap(output), threshold)
    return output


def pool(input, rates, kernel, stride=None, pad=0):
    """
    Apply 2D max pooling

    Parameters
    ----------
    input : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Pooling input tensor
    kernel : int or list of 2 ints
        Kernel size of the pooling
    stride : None or int or list of 2 ints, optional
        Convolution stride. If stride is set to None it will be the same as kernel (default None)
    pad : int or list of 2 ints or list of 4 ints, optional
        Padding size of the input (default 0)

    Returns
    -------
    spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Pooling output tensor
    """

    if stride is None:
        stride = kernel
    kernel = expand2(kernel)
    stride = expand2(stride)
    pad = expand4(pad)

    input_ = to5(wrap(input))
    shape = impl.shape.pool(input_.shape, kernel, stride, pad)
    output = create(input, input_.dtype, shape)
    impl.rate.pool(input_, wrap(rates), wrap(output), kernel, stride, pad)
    return output
