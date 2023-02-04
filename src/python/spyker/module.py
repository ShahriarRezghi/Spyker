from spyker.utils import *
import spyker.spyker_plugin as impl


class DoGFilter(impl.DoGFilter):
    """
    Parameter holder for DoG filter

    Attributes
    ----------
    std1 : float
        Standard deviation of the first Gaussian filter
    std2 : float
        Standard deviation of the second Gaussian filter
    """

    def __init__(self, std1, std2):
        """
        Parameters
        ----------
        std1 : float
            Standard deviation of the first Gaussian filter
        std2 : float
            Standard deviation of the second Gaussian filter
        """

        super().__init__(std1, std2)


class GaborFilter(impl.GaborFilter):
    """
    Parameter holder for Gabor filter

    Attributes
    ----------
    sigma : float
        Standard deviation, width of the strips of the filter
    theta : float
        Orientation, orientation of the filter
    gamma : float
        Spatial aspect ratio, height of the stripes, reverse relation
    lambda_ : float
        Wavelength, spacing between strips, reverse relation
    psi : float
        Phase offset, spacial shift of the strips
    """

    def __init__(self, sigma, theta, gamma, lamda, psi):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation
        theta : float
            Orientation
        gamma : float
            Spatial aspect ratio
        lambda_ : float
            Wavelength
        psi : float
            Phase offset
        """
        
        super().__init__(sigma, theta, gamma, lamda, psi)


class STDPConfig(impl.STDPConfig):
    """
    Parameter holder for STDP configuration

    Attributes
    ----------
    positive : float
        Positive learning rate
    negative : float
        Negative learning rate
    stabilize : bool
        Stabilization
    lower : float
        Lower bound of the weights
    upper : float
        Upper bound of the weights
    """

    def __init__(self, positive, negative, stabilize=True, lower=0, upper=1):
        """
        Parameters
        ----------
        positive : float
            Positive learning rate
        negative : float
            Negative learning rate
        stabilize : bool, optional
            Stabilization (default True)
        lower : float, optional
            Lower bound of the weights (default 0.0)
        upper : float, optional
            Upper bound of the weights (default 1.0)
        """

        super().__init__(positive, negative, stabilize, lower, upper)


class BPConfig(impl.BPConfig):
    def __init__(self, sfactor, lrate, lrf, lamda):
        super().__init__(sfactor, lrate, lrf, lamda)


class ZCA(impl.ZCA):
    """
    ZCA Whitening Module

    Attributes
    ----------
    mean : spyker.tensor
        Mean values of features
    transform : spyker.tensor
        Whitening transformation matrix

    Methods
    -------
    fit(input, epsilon, trnasorm=False):
        Fit the input data
    __call__(*X, inplace=True):
        Transform the input data
    split(input):
        Split ZCA transformed data into two positive and negative channels
    """

    def __init__(self):
        super().__init__()

    def fit(self, array, epsilon, transform=False):
        """
        Fit the input data

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input data to be fitted to
        epsilon : float
            Epsilon parameter of the trnasformation
        trnasform : bool
            Whether to transform the input inplace or not (default False)

        Returns
        -------
        spyker.ZCA
            this class
        """

        self._fit(wrap(array), epsilon, transform)
        return self

    def __call__(self, *array, inplace=True):
        """
        Transform the input data

        Parameters
        ----------
        X : Any number of spyker.tensor or torch.tensor or numpy.ndarray
            Input data to be transformed
        inplace : bool
            Whether to transform the input inplace or not (default True).

        Returns
        -------
        Any number of spyker.tensor or torch.tensor or numpy.ndarray
            Transformed data
        """

        if not inplace: array = [copy(x) for x in array]
        [self._forward(wrap(x)) for x in array]
        return tuple(array) if len(array) > 1 else array[0]

    @staticmethod
    def split(array):
        """
        Split ZCA transformed data into two positive and negative channels

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input data to be splitted

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            splitted data
        """

        input_ = least2(wrap(array))
        shape = impl.shape.zca_split(input_.shape)
        output = create(array, input_.dtype, shape)
        impl.ZCA._split(input_, wrap(output))
        return output

    def save(self, path):
        """
        Save the model to a file

        Parameters
        ----------
        path : string
            Path of the file to be saved into
        """

        mean = spyker.to_numpy(self.mean)
        transform = spyker.to_numpy(self.transform)
        numpy.savez(path, mean=mean, transform=transform)

    def load(self, path):
        """
        Load the model to a file

        Parameters
        ----------
        path : string
            Path of the file to be loaded from
        """

        data = numpy.load(path)
        wrap(data['mean']).to(self.mean)
        wrap(data['transform']).to(self.transform)


class DoG(impl.DoG):
    """
    2D difference of Gaussian (DoG) filter module

    Attributes
    ----------
    kernel : spyker.tensor
        Kernel of the filter

    Methods
    -------
    __call__(input)
        Apply the filter on the input
    """

    def __init__(self, size, filters, pad=0, device=impl.device('cpu'), dtype='f32'):
        """
        Parameters
        ----------
        size : int
            Half size of the window. full size of the window is 2 * size + 1
        filters : spyker.DoGFilter or list of 'spyker.DoGFilter's
            List of filters to be applied
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        pad = expand4(pad)
        if isinstance(filters, DoGFilter): filters = [filters]
        super().__init__(device, size, filters, pad, dtype)
        self.device = device
        self.pad = pad

    def __call__(self, array):
        """
        Apply the filter on the input

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Filtered output tensor
        """

        input_ = to4(wrap(array))
        shape = impl.shape.dog(input_.shape, self.kernel.shape, self.pad)
        output = create(array, self.kernel.dtype, shape)
        self._forward(input_, wrap(output))
        return output


class Gabor(impl.Gabor):
    """
    2D Gabor filter module

    Attributes
    ----------
    kernel : spyker.tensor
        Kernel of the filter

    Methods
    -------
    __call__(input)
        Apply the filter on the input
    """

    def __init__(self, size, filters, pad=0, device=impl.device('cpu'), dtype='f32'):
        """
        Parameters
        ----------
        size : int
            Half size of the window. full size of the window is 2 * size + 1
        filters : spyker.GaborFilter or list of 'spyker.GaborFilter's
            List of filters to be applied
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        pad = expand4(pad)
        if isinstance(filters, GaborFilter): filters = [filters]
        super().__init__(device, size, filters, pad, dtype)
        self.device = device
        self.pad = pad

    def __call__(self, array):
        """
        Apply the filter on the input

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Filtered output tensor
        """

        input_ = to4(wrap(array))
        shape = impl.shape.gabor(input_.shape, self.kernel.shape, self.pad)
        output = create(array, self.kernel.dtype, shape)
        self._forward(input_, wrap(output))
        return output


class LoG(impl.LoG):
    """
    2D Laplacian of Gaussian (LoG) filter module

    Attributes
    ----------
    kernel : spyker.tensor
        Kernel of the filter

    Methods
    -------
    __call__(input)
        Apply the filter on the input
    """

    def __init__(self, size, stds, pad=0, device=impl.device('cpu'), dtype='f32'):
        """
        Parameters
        ----------
        size : int
            Half size of the window. full size of the window is 2 * size + 1
        stds : list of floats
            List of stds of LoG filters to be applied
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        pad = expand4(pad)
        super().__init__(device, size, stds, pad, dtype)
        self.device = device
        self.pad = pad

    def __call__(self, array):
        """
        Apply the filter on the input

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Filtered output tensor
        """

        input_ = to4(wrap(array))
        shape = impl.shape.log(input_.shape, self.kernel.shape, self.pad)
        output = create(array, self.kernel.dtype, shape)
        self._forward(input_, wrap(output))
        return output


class FC(impl.FC):
    """
    Fully connected module

    Attributes
    ----------
    kernel : spyker.tensor
        Kernel of the module
    config : list of 'STDPConfig's
        List of STDP configurations

    Methods
    -------
    __call__(input)
        Apply the fully connected on the input
    stdp(array, winners, output)
        Apply the STDP on the fully connected
    """

    def __init__(self, insize, outsize, mean=.5, std=.02, device=impl.device('cpu'), dtype='f32'):
        """
        Parameters
        ----------
        insize : int
            Dimensions of the input signal
        outsize : int
            Dimensions of the output signal
        mean : float, optional
            Mean of the random normal variable that initializes the kernel (default 0.5)
        std : float, optional
            Standard deviation of the random normal variable that initializes the kernel (default 0.02)
        """

        super().__init__(device, insize, outsize, mean, std, dtype)
        self.device = device
    
    def __call__(self, array, sign=False):
        """
        Apply the fully connected on the input

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input dense tensor to be processed

        Returns
        -------
        output : spyker.tensor or torch.tensor or numpy.ndarray
            Output dense tensor to be written to
        """

        input_ = to3(wrap(array))
        shape = impl.shape.fc(input_.shape, self.kernel.shape)
        output = create(array, self.kernel.dtype, shape)
        self._forward(input_, wrap(output), sign)
        return output

    def stdp(self, array, winners, output):
        """
        Apply the STDP on the fully connected

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Fully connected input dense tensor
        winners : list of list of 'spyker.Winner's
            Winner neurons that are selected for updating
        output : spyker.tensor or torch.tensor or numpy.ndarray
            Fully connected output dense tensor
        """

        self._stdp(wrap(array), winners, wrap(output))

    def backward(self, array, output, grad):
        input_, grad_ = to2(wrap(array)), to2(wrap(grad))
        next = create(grad, grad_.dtype, input_.shape)
        self._backward(input_, wrap(output), grad_, wrap(next))
        return next


class Conv(impl.Conv):
    """
    2D convolution module

    Attributes
    ----------
    kernel : spyker.tensor
        Kernel of the module
    config : list of 'STDPConfig's
        List of STDP configurations

    Methods
    -------
    __call__(input)
        Apply the convolution on the input
    """

    def __init__(self, insize, outsize, kernel, stride=1, pad=0, mean=.5, std=.02, device=impl.device('cpu'), dtype='f32'):
        """
        Parameters
        ----------
        insize : int
            Channels of the input signal
        outsize : int
            Channels of the output signal
        kernel : int or list of 2 ints
            Kernel size of the convolution
        stride : int or list of 2 ints
            Size of the convolution (default 1)
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0).
        mean : float
            Mean of the random normal variable that initializes the kernel (default 0.5)
        std : float
            Standard deviation of the random normal variable that initializes the kernel (default 0.02)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        kernel = expand2(kernel)
        stride = expand2(stride)
        pad = expand4(pad)

        super().__init__(device, insize, outsize, kernel, stride, pad, mean, std, dtype)
        self.device = device
        self.stride = stride
        self.pad = pad

    def __call__(self, array, threshold=None):
        """
        Apply the convolution on the input

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Convolved output tensor
        """

        if threshold is not None:
            return self._forward(array, threshold)

        input_ = to5(wrap(array))
        shape = impl.shape.conv(input_.shape, self.kernel.shape, self.stride, self.pad)
        output = create(array, self.kernel.dtype, shape)
        self._forward(input_, wrap(output))
        return output

    def stdp(self, array, winners, output=None):
        """
        Apply STDP to update the weights of convolution

        Parameters
        ----------
        array : spyker.tensor or torch.tensor or numpy.ndarray
            Convolution input dense tensor
        winners : list of list of 'spyker.Winner's
            Winner neurons that are selected for updating
        output : spyker.tensor or torch.tensor or numpy.ndarray
            Convolution output dense tensor
        """

        if output is None:
            self._stdp(array, winners)
        else:
            self._stdp(wrap(array), winners, wrap(output))


def canny(array, low, high):
    """
    Apply Canny edge detection

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Edge detection input dense tensor
    low : float
        Threshold for weak edges
    high : float
        threshold for strong edges

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Edge detection output tensor
    """

    input_ = to4(wrap(array))
    output = create(array, 'u8', input_.shape)
    impl.canny(input_, wrap(output), low, high)
    return output


def fc(array, kernel, sign=False):
    """
    Apply fully connected operation

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected input dense tensor
    kernel : spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected kernel dense tensor

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected output dense tensor
    """

    input_ = to3(wrap(array))
    kernel_ = wrap(kernel)
    shape = impl.shape.fc(input_.shape, kernel_.shape)
    output = create(array, kernel_.dtype, shape)
    impl.fc(input_, kernel_, wrap(output), sign)
    return output


def conv(array, kernel, stride=1, pad=0):
    """
    Apply 2D convolution

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Convolution input dense tensor
    kernel : spyker.tensor or torch.tensor or numpy.ndarray
        Convolution kernel dense tensor
    stride : int or list of 2 ints, optional
        Convolution stride (default 1)
    pad : int or list of 2 ints or list of 4 ints, optional
        Padding size of the input (default 0)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Convolved output dense tensor
    """

    kernel_ = wrap(kernel)
    stride = expand2(stride)
    pad = expand4(pad)

    input_ = to5(wrap(array))
    shape = impl.shape.conv(input_.shape, kernel_.shape, stride, pad)
    output = create(array, kernel_.dtype, shape)
    impl.conv(input_, kernel_, wrap(output), stride, pad)
    return output


def pad(array, pad, value=0):
    """
    2D padding

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Input dense tensor
    pad : int or list of 2 ints or list of 4 ints, optional
        Padding size of the input (default 0)
    value : float
        Padding value (default 0.0)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Padding output dense tensor
    """
    
    pad = expand4(pad)
    
    input_ = to5(wrap(array))
    shape = impl.shape.pad(input_.shape, pad)
    output = create(array, input_.dtype, shape)
    impl.pad(input_, wrap(output), pad, value)
    return output


def threshold(array, threshold, value=.0, inplace=True):
    """
    Apply thresholding

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input tensor
    threshold : float
        Threshold of firing
    value : float, optional
        Value to replace (default 0.0)
    inplace : bool, optional
        Override the input or not (default True)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        If inplace input tensor, otherwise output tensor
    """

    if not inplace: array = copy(array)
    impl.threshold(wrap(array, write=True), threshold, value)
    return array


def quantize(array, lower, middle, upper, inplace=True):
    """
    Quantize input

    If values are low than mid then it is set to low and set to high otherwise

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input dense tensor to be quantized
    lower : float
        Lower value to be set
    middle : float
        Middle value to be compared
    upper : float
        Upper value to be set
    inplace : bool, optional
        Override the input or not (default True)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        If inplace input tensor, otherwise output tensor
    """

    if not inplace: array = copy(array)
    impl.quantize(wrap(array, write=True), lower, middle, upper)
    return array


def code(array, time, sort=True, dtype='u8', code='rank'):
    """
    Apply rank coding

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input dense tensor
    time : int
        Number of time steps
    sort : bool, optional
        Whether to sort the values or not (default True). Sorting might increase accuracy but it deceases performance

    Returns
    -------
    spyker.Tensor or torch.tensor or numpy.ndarray
        Rank coding output tensor
    """

    input_ = least2(wrap(array))
    shape = impl.shape.code(input_.shape, time)
    output = create(array, dtype, shape)
    impl.code(input_, wrap(output), time, sort, code)
    return output


def infinite(array, value=.0, inplace=True):
    """
    Apply infinite thresholding

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input tensor
    value : float, optional
        value to replace (default 0.0)
    inplace : bool, optional
        Override the input or not (default True)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        The input tensor
    """

    if not inplace: array = copy(array)
    input_ = least3(wrap(array, write=True))
    impl.infinite(input_, value)
    return array


def fire(array, threshold=.0, dtype='u8', code='rank'):
    """
    Apply integrate-and-fire mechanism

    If the input is already thresholded then there is no need to pass threshold to this function.

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input tensor
    threshold : float, optional
        Threshold of firing (default 0)
    dtype : spyker data type
        Data type of the created output tensor (default spyker.u8)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Integrate-and-fire output tensor
    """

    input_ = wrap(array)
    output = create(array, dtype, input_.shape)
    impl.fire(input_, wrap(output), threshold, code)
    return output


def gather(array, threshold=.0, dtype='u8', code='rank'):
    """
    Gather temporal information

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input tensor
    threshold : float, optional
        Threshold of firing (default 0.0)
    dtype : spyker data type
        Data type of the created output tensor (default spyker.u8)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Gathered output tensor
    """

    input_ = least3(wrap(array))
    output = create(array, dtype, impl.shape.gather(input_.shape))
    impl.gather(input_, wrap(output), threshold, code)
    return output


def scatter(array, time, dtype='u8'):
    """
    Scatter temporal information

    Parameters
    ----------
    array : spyker.tensor or torch.tensor or numpy.ndarray
        Input tensor
    time : int
        Number of time steps to scatter the input to
    dtype : spyker data type
        Data type of the created output tensor (default spyker.u8)

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Scattered output tensor
    """

    input_ = least2(wrap(array))
    output = create(array, dtype, impl.shape.scatter(input_.shape, time))
    impl.scatter(input_, wrap(output))
    return output


def pool(array, kernel, stride=None, pad=0, rates=None):
    """
    Apply 2D max pooling

    Parameters
    ----------
    array : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
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

    input_ = to5(wrap(array))
    shape = impl.shape.pool(input_.shape, kernel, stride, pad)
    output = create(array, input_.dtype, shape)
    rates_ = impl.tensor() if rates is None else wrap(rates)
    impl.pool(input_, wrap(output), kernel, stride, pad, rates_)
    return output


def inhibit(array, threshold=.0, inplace=True):
    """
    Apply lateral Inhibition (inplace)

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    array : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Input tensor
    threshold : float, optional
        Threshold of firing (default 0.0). Only used when input has type F32
    inplace : bool, optional
        Override the input or not

    Returns
    -------
    spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        If inplace input tensor, otherwise output tensor
    """

    if not inplace: array = copy(array)
    impl.inhibit(wrap(array, write=True), threshold)
    return array


def fcwta(array, radius, count, threshold=.0):
    return impl.fcwta(wrap(array), radius, count, threshold)


def convwta(array, radius, count, threshold=.0):
    """
    Select winner neurons

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    array : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
        Input tensor
    radius : int or list of 2 ints
        Radius of inhibition
    threshold : float, optional
        Threshold of firing (default 0.0). Only used when input has type F32

    Returns
    -------
    list of list of 'spyker.Winner's
        Winner neurons
    """

    radius = expand2(radius)
    return impl.convwta(wrap(array), radius, count, threshold)


def stdp(conv, array, winners, output):
    impl.stdp(conv.impl, array, winners, output)


def backward(array, target, time, gamma):
    input_ = wrap(array)
    output = create(array, input_.dtype, input_.shape)
    impl.backward(input_, wrap(output), wrap(target), time, gamma)
    return output


def labelize(array, threshold):
    input_ = wrap(array)
    output = create(array, 'i64', [input_.shape[0]])
    impl.labelize(input_, wrap(output), threshold)
    return output
