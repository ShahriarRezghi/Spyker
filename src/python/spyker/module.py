from spyker.utils import *
import spyker.spyker_plugin as impl


class DoGFilter:
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

        self.std1 = std1
        self.std2 = std2

    def get(self):
        return impl.DoGFilter(self.std1, self.std2)


class GaborFilter:
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

    def __init__(self, sigma, theta, gamma, lambda_, psi):
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
        
        self.sigma = sigma
        self.theta = theta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.psi = psi

    def get(self):
        return impl.GaborFilter(self.sigma, self.theta, self.gamma, self.lambda_, self.psi)


class STDPConfig:
    """
    Parameter holder for STDP configuration

    Attributes
    ----------
    pos : float
        Positive learning rate
    neg : float
        Negative learning rate
    stabilize : bool
        Stabilization
    low : float
        Lower bound of the weights
    high : float
        Upper bound of the weights
    """

    def __init__(self, pos, neg, stabilize=True, low=0, high=1):
        """
        Parameters
        ----------
        pos : float
            Positive learning rate
        neg : float
            Negative learning rate
        stabilize : bool, optional
            Stabilization (default True)
        low : float, optional
            Lower bound of the weights (default 0.0)
        high : float, optional
            Upper bound of the weights (default 1.0)
        """

        self.pos = pos
        self.neg = neg
        self.stabilize = stabilize
        self.low = low
        self.high = high

    def get(self):
        return impl.STDPConfig(self.pos, self.neg, self.stabilize, self.low, self.high)


class BPConfig:
    def __init__(self, sfactor, lrate, lrf, lambda_):
        self.sfactor = sfactor
        self.lrate = lrate
        self.lrf = lrf
        self.lambda_ = lambda_

    def get(self):
        return impl.BPConfig(self.sfactor, self.lrate, self.lrf, self.lambda_)


class ZCA:
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
        self.impl = impl.ZCA()

    @property
    def mean(self):
        return self.impl.mean

    @mean.setter
    def mean(self, mean):
        self.impl.mean = mean

    @property
    def transform(self):
        return self.impl.transform

    transform.setter
    def transform(self, transform):
        self.impl.transform = transform

    def fit(self, input, epsilon, transform=False):
        """
        Fit the input data

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
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

        self.impl.fit(wrap(input), epsilon, transform)
        return self

    def __call__(self, *X, inplace=True):
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

        output = []
        for input in X:
            input = input if inplace else copy(input)
            self.impl(wrap(input))
            output.append(input)
        return tuple(output) if len(output) > 1 else output[0]

    @staticmethod
    def split(input):
        """
        Split ZCA transformed data into two positive and negative channels

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Input data to be splitted

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            splitted data
        """

        input_ = least2(wrap(input))
        shape = impl.shape.zca_split(input_.shape)
        output = create(input, input_.dtype, shape)
        impl.ZCA.split(input_, wrap(output))
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
        trans = spyker.to_numpy(self.trans)
        numpy.savez(path, mean=mean, trans=trans)

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
        wrap(data['trans']).to(self.trans)

class DoG:
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

    def __init__(self, size, filter, pad=0, device=impl.device('cpu')):
        """
        Parameters
        ----------
        size : int
            Half size of the window. full size of the window is 2 * size + 1
        filter : spyker.DoGFilter or list of 'spyker.DoGFilter's
            List of filters to be applied
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        pad = expand4(pad)
        if isinstance(filter, DoGFilter): filter = [filter]
        filter = [x.get() for x in filter]
        self.impl = impl.DoG(device, size, filter, pad)
        self.device = device
        self.pad = pad

    @property
    def kernel(self):
        return self.impl.kernel
    
    @kernel.setter
    def kernel(self, kernel):
        self.impl.kernel = kernel

    def __call__(self, input):
        """
        Apply the filter on the input

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Filtered output tensor
        """

        input_ = to4(wrap(input))
        shape = impl.shape.dog(input_.shape, self.kernel.shape, self.pad)
        output = create(input, impl.f32, shape)
        self.impl(input_, wrap(output))
        return output


class Gabor:
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

    def __init__(self, size, filter, pad=0, device=impl.device('cpu')):
        """
        Parameters
        ----------
        size : int
            Half size of the window. full size of the window is 2 * size + 1
        filter : spyker.GaborFilter or list of 'spyker.GaborFilter's
            List of filters to be applied
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        pad = expand4(pad)
        if isinstance(filter, GaborFilter): filter = [filter]
        filter = [x.get() for x in filter]
        self.impl = impl.Gabor(device, size, filter, pad)
        self.device = device
        self.pad = pad

    @property
    def kernel(self):
        return self.impl.kernel

    @kernel.setter
    def kernel(self, kernel):
        self.impl.kernel = kernel

    def __call__(self, input):
        """
        Apply the filter on the input

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Filtered output tensor
        """

        input_ = to4(wrap(input))
        shape = impl.shape.gabor(input_.shape, self.kernel.shape, self.pad)
        output = create(input, impl.f32, shape)
        self.impl(input_, wrap(output))
        return output


class LoG:
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

    def __init__(self, size, std, pad=0, device=impl.device('cpu')):
        """
        Parameters
        ----------
        size : int
            Half size of the window. full size of the window is 2 * size + 1
        std : list of floats
            List of stds of LoG filters to be applied
        pad : int or list of 2 ints or list of 4 ints, optional
            Padding size of the input (default 0)
        device : spyker.device, optional
            Device of the filter to be run on (default CPU)
        """

        pad = expand4(pad)
        self.impl = impl.LoG(device, size, std, pad)
        self.device = device
        self.pad = pad

    @property
    def kernel(self):
        return self.impl.kernel
    
    @kernel.setter
    def kernel(self, kernel):
        self.impl.kernel = kernel

    def __call__(self, input):
        """
        Apply the filter on the input

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Filtered output tensor
        """

        input_ = to4(wrap(input))
        shape = impl.shape.log(input_.shape, self.kernel.shape, self.pad)
        output = create(input, impl.f32, shape)
        self.impl(input_, wrap(output))
        return output


class FC:
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
    stdp(input, winners, output)
        Apply the STDP on the fully connected
    """

    def __init__(self, input, output, mean=.5, std=.02, device=impl.device('cpu')):
        """
        Parameters
        ----------
        input : int
            Dimensions of the input signal
        output : int
            Dimensions of the output signal
        mean : float, optional
            Mean of the random normal variable that initializes the kernel (default 0.5)
        std : float, optional
            Standard deviation of the random normal variable that initializes the kernel (default 0.02)
        """

        self.impl = impl.FC(device, input, output, mean, std)
        self.device = device
    
    @property
    def config(self):
        return self.impl.config

    @property
    def kernel(self):
        return self.impl.kernel

    @property
    def bpconfig(self):
        return self.impl.bpconfig

    @kernel.setter
    def kernel(self, kernel):
        self.impl.kernel = kernel

    @config.setter
    def config(self, config):
        if isinstance(config, STDPConfig):
            self.impl.config = [config.get()]
        else: 
            self.impl.config = [x.get() for x in config]

    @bpconfig.setter
    def bpconfig(self, config):
        self.impl.bpconfig = config.get()

    def __call__(self, input):
        """
        Apply the fully connected on the input

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Input dense tensor to be processed

        Returns
        -------
        output : spyker.tensor or torch.tensor or numpy.ndarray
            Output dense tensor to be written to
        """

        input_ = to3(wrap(input))
        shape = impl.shape.fc(input_.shape, self.kernel.shape)
        output = create(input, impl.f32, shape)
        self.impl(input_, wrap(output))
        return output

    def stdp(self, input, winners, output):
        """
        Apply the STDP on the fully connected

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Fully connected input dense tensor
        winners : list of list of 'spyker.Winner's
            Winner neurons that are selected for updating
        output : spyker.tensor or torch.tensor or numpy.ndarray
            Fully connected output dense tensor
        """

        self.impl.stdp(wrap(input), winners, wrap(output))

    def backward(self, input, output, grad):
        input_, grad_ = to2(wrap(input)), to2(wrap(grad))
        next = create(grad, grad_.dtype, input_.shape)
        self.impl.backward(input_, wrap(output), grad_, wrap(next))
        return next


class Conv:
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

    def __init__(self, input, output, kernel, stride=1, pad=0, mean=.5, std=.02, device=impl.device('cpu')):
        """
        Parameters
        ----------
        input : int
            Channels of the input signal
        output : int
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

        self.impl = impl.Conv(device, input, output, kernel, stride, pad, mean, std)
        self.device = device
        self.stride = stride
        self.pad = pad

    @property
    def kernel(self):
        return self.impl.kernel

    @property
    def config(self):
        return self.impl.config

    @kernel.setter
    def kernel(self, kernel):
        self.impl.kernel = kernel

    @config.setter
    def config(self, config):
        if isinstance(config, STDPConfig):
            self.impl.config = [config.get()]
        else: 
            self.impl.config = [x.get() for x in config]


    def __call__(self, input):
        """
        Apply the convolution on the input

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Input tensor to be processed

        Returns
        -------
        spyker.tensor or torch.tensor or numpy.ndarray
            Convolved output tensor
        """

        input_ = to5(wrap(input))
        shape = impl.shape.conv(input_.shape, self.kernel.shape, self.stride, self.pad)
        output = create(input, impl.f32, shape)
        self.impl(input_, wrap(output))
        return output

    def stdp(self, input, winners, output):
        """
        Apply STDP to update the weights of convolution

        Parameters
        ----------
        input : spyker.tensor or torch.tensor or numpy.ndarray
            Convolution input dense tensor
        winners : list of list of 'spyker.Winner's
            Winner neurons that are selected for updating
        output : spyker.tensor or torch.tensor or numpy.ndarray
            Convolution output dense tensor
        """

        self.impl.stdp(wrap(input), winners, wrap(output))


def fcfwd(fc, input):
    """
    Apply fully connected

    Parameters
    ----------
    fc : spyker.FC
        Fully connected class to use
    input : spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected input dense tensor

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected output dense tensor
    """

    input_ = to3(wrap(input))
    shape = impl.shape.fc(input_.shape, fc.kernel.shape)
    output = create(input, impl.f32, shape)
    impl.fc(fc.impl, input_, wrap(output))
    return output


def convfwd(conv, input):
    """
    Apply 2D Convolution

    Parameters
    ----------
    conv : spyker.Conv
        Convolution class to use
    input : spyker.tensor or torch.tensor or numpy.ndarray
        Convolution input dense tensor

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Convolution output dense tensor
    """

    input_ = to5(wrap(input))
    shape = impl.shape.conv(input_.shape, conv.kernel.shape, conv.stride, conv.pad)
    output = create(input, impl.f32, shape)
    impl.conv(conv.impl, input_, wrap(output))
    return output


def canny(input, low, high):
    """
    Apply Canny edge detection

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    input_ = to4(wrap(input))
    output = create(input, impl.u8, input_.shape)
    impl.canny(input_, wrap(output), low, high)
    return output


def fc(input, kernel):
    """
    Apply fully connected operation

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected input dense tensor
    kernel : spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected kernel dense tensor

    Returns
    -------
    spyker.tensor or torch.tensor or numpy.ndarray
        Fully connected output dense tensor
    """

    input_ = to3(wrap(input))
    kernel_ = wrap(kernel)
    shape = impl.shape.fc(input_.shape, kernel_.shape)
    output = create(input, impl.f32, shape)
    impl.fc(input_, kernel_, wrap(output))
    return output


def conv(input, kernel, stride=1, pad=0):
    """
    Apply 2D convolution

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray or spyker.Sparse
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

    input_ = to5(wrap(input))
    shape = impl.shape.conv(input_.shape, kernel_.shape, stride, pad)
    output = create(input, impl.f32, shape)
    impl.conv(input_, kernel_, wrap(output), stride, pad)
    return output


def pad(input, pad, value=0):
    """
    2D padding

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray or spyker.Sparse
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
    
    input_ = to5(wrap(input))
    shape = impl.shape.pad(input_.shape, pad)
    output = create(input, input_.dtype, shape)
    impl.pad(input_, wrap(output), pad, value)
    return output


def threshold(input, threshold, value=.0, inplace=True):
    """
    Apply thresholding

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    if not inplace: input = copy(input)
    impl.threshold(wrap(input, write=True), threshold, value)
    return input


def quantize(input, lower, middle, upper, inplace=True):
    """
    Quantize input

    If values are low than mid then it is set to low and set to high otherwise

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    if not inplace: input = copy(input)
    impl.quantize(wrap(input, write=True), lower, middle, upper)
    return input


def code(input, time, sort=True, dtype=impl.u8):
    """
    Apply rank coding

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    input_ = least2(wrap(input))
    shape = impl.shape.code(input_.shape, time)
    output = create(input, dtype, shape)
    impl.code(input_, wrap(output), time, sort)
    return output


def infinite(input, value=.0, inplace=True):
    """
    Apply infinite thresholding

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    if not inplace: input = copy(input)
    input_ = least3(wrap(input, write=True))
    impl.infinite(input_, value)
    return input


def fire(input, threshold=.0, dtype=impl.u8):
    """
    Apply integrate-and-fire mechanism

    If the input is already thresholded then there is no need to pass threshold to this function.

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    input_ = wrap(input)
    output = create(input, dtype, input_.shape)
    impl.fire(input_, wrap(output), threshold)
    return output


def gather(input, threshold=.0, dtype=impl.u8):
    """
    Gather temporal information

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    input_ = least3(wrap(input))
    output = create(input, dtype, impl.shape.gather(input_.shape))
    impl.gather(input_, wrap(output), threshold)
    return output


def scatter(input, time, dtype=impl.u8):
    """
    Scatter temporal information

    Parameters
    ----------
    input : spyker.tensor or torch.tensor or numpy.ndarray
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

    input_ = least2(wrap(input))
    output = create(input, dtype, impl.shape.scatter(input_.shape, time))
    impl.scatter(input_, wrap(output))
    return output


def pool(input, kernel, stride=None, pad=0):
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
    impl.pool(input_, wrap(output), kernel, stride, pad)
    return output


def inhibit(input, threshold=.0, inplace=True):
    """
    Apply lateral Inhibition (inplace)

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    input : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
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

    if not inplace: input = copy(input)
    impl.inhibit(wrap(input, write=True), threshold)
    return input


def fcwta(input, radius, count, threshold=.0):
    return impl.fcwta(wrap(input), radius, count, threshold)


def convwta(input, radius, count, threshold=.0):
    """
    Select winner neurons

    If the input is already thresholded or sparse then there is no need to pass threshold to this function.

    Parameters
    ----------
    input : spyker.Tensor or torch.tensor or numpy.ndarray or spyker.Sparse
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
    return impl.convwta(wrap(input), radius, count, threshold)


def stdp(conv, input, winners, output):
    impl.stdp(conv.impl, input, winners, output)


def backward(input, target, time, gamma):
    input_ = wrap(input)
    output = create(input, input_.dtype, input_.shape)
    impl.backward(input_, wrap(output), wrap(target), time, gamma)
    return output


def labelize(input, threshold):
    input_ = wrap(input)
    output = create(input, impl.i64, [input_.shape[0]])
    impl.labelize(input_, wrap(output), threshold)
    return output


def signfc(input, kernel):
    input_ = to3(wrap(input))
    kernel_ = wrap(kernel)
    shape = impl.shape.fc(input_.shape, kernel_.shape)
    output = create(input, impl.f32, shape)
    impl.signfc(input_, kernel_, wrap(output))
    return output


def signfcfwd(fc, input):
    input_ = to3(wrap(input))
    shape = impl.shape.fc(input_.shape, fc.kernel.shape)
    output = create(input, impl.f32, shape)
    impl.signfc(fc.impl, input_, wrap(output))
    return output
