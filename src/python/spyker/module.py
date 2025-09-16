# BSD 3-Clause License
#
# Copyright (c) 2022-2025, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import spyker.spyker_plugin as impl
from spyker.private import expand2, expand4, least2, least3, to2, to3, to4, to5
from spyker.utils import CodingType, DataType, TensorLike, clone_array, create_array, to_numpy, wrap_array


class DoGFilter(impl.DoGFilter):
    r"""
    Difference-of-Gaussians (DoG) filter parameters.

    A lightweight parameter holder that mirrors the C++ DoG filter config.

    Attributes
    ----------
    std1 : float
        Standard deviation of the first Gaussian.
    std2 : float
        Standard deviation of the second Gaussian.
    """

    def __init__(self, std1: float, std2: float) -> None:
        """
        Parameters
        ----------
        std1 : float
            Standard deviation of the first Gaussian.
        std2 : float
            Standard deviation of the second Gaussian.
        """
        super().__init__(std1, std2)


class GaborFilter(impl.GaborFilter):
    r"""
    Gabor filter parameters.

    Encapsulates the 2D Gabor kernel configuration commonly used for
    orientation- and frequency-selective filtering.

    Attributes
    ----------
    sigma : float
        Standard deviation (controls envelope size / bandwidth).
    theta : float
        Orientation angle in radians (filter rotation).
    gamma : float
        Spatial aspect ratio (y/x); controls stripe elongation.
    lambda_ : float
        Wavelength (spatial period) of the sinusoidal carrier.
    psi : float
        Phase offset of the carrier in radians.
    """

    def __init__(self, sigma: float, theta: float, gamma: float, lambda_: float, psi: float) -> None:
        """
        Parameters
        ----------
        sigma : float
            Standard deviation (envelope width).
        theta : float
            Orientation angle in radians.
        gamma : float
            Spatial aspect ratio (y/x).
        lambda_ : float
            Wavelength (spatial period).
        psi : float
            Phase offset in radians.
        """
        super().__init__(sigma, theta, gamma, lambda_, psi)


class STDPConfig(impl.STDPConfig):
    r"""
    Spike-Timing-Dependent Plasticity (STDP) configuration.

    Attributes
    ----------
    positive : float
        Potentiation learning rate (pre→post timing).
    negative : float
        Depression learning rate (post→pre timing).
    stabilize : bool
        Enable weight stabilization.
    lower : float
        Lower bound for synaptic weights.
    upper : float
        Upper bound for synaptic weights.
    """

    def __init__(
        self,
        positive: float,
        negative: float,
        stabilize: bool = True,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        positive : float
            Potentiation learning rate.
        negative : float
            Depression learning rate.
        stabilize : bool, default=True
            Whether to enable weight stabilization.
        lower : float, default=0.0
            Lower bound for synaptic weights.
        upper : float, default=1.0
            Upper bound for synaptic weights.
        """
        super().__init__(positive, negative, stabilize, lower, upper)


class BPConfig(impl.BPConfig):
    r"""
    Backpropagation (BP) configuration.

    Holds scalar hyperparameters used by Spyker's backprop-based learners.

    Parameters
    ----------
    sfactor : float
        Scaling factor applied to the loss/gradients (implementation-dependent).
    lrate : float
        Base learning rate.
    lrf : float
        Learning-rate falloff/decay factor (e.g., multiplicative schedule).
    lamda : float
        Regularization coefficient (e.g., L2 weight decay).

    Notes
    -----
    This class mirrors the underlying C++ ``BPConfig`` and is a thin parameter wrapper.
    """

    def __init__(self, sfactor: float, lrate: float, lrf: float, lamda: float) -> None:
        super().__init__(sfactor, lrate, lrf, lamda)


class DoG(impl.DoG):
    r"""
    2D Difference-of-Gaussians (DoG) filtering module.

    Wraps the C++ DoG operator and manages padding, device/dtype placement,
    and zero-copy interop with NumPy/PyTorch/Spyker tensors.

    Attributes
    ----------
    kernel : spyker.Tensor
        The assembled convolution kernel bank (read-only from Python).
    device : spyker.Device
        Device where the module's parameters reside (e.g., CPU or CUDA device).
    """

    def __init__(
        self,
        size: int,
        filters: Union[DoGFilter, Sequence[DoGFilter]],
        pad: Union[int, Sequence[int]] = 0,
        device: impl.Device | None = None,
        dtype: DataType = "f32",
    ) -> None:
        """
        Parameters
        ----------
        size : int
            Half window size; the full kernel size is ``2 * size + 1``.
        filters : DoGFilter or sequence of DoGFilter
            One or more DoG filter parameter sets to instantiate the kernel bank.
        pad : int or sequence of 2 or 4 ints, default=0
            Input padding. If an int, symmetric padding is applied to both axes.
            A sequence of 2 is interpreted as (width, height); a sequence of 4 as
            (left, right, top, bottom).
        device : impl.Device or None, default=None
            Destination device for the module parameters and computations (default CPU).
        dtype : DataType, default="f32"
            Scalar dtype code for parameters/outputs (Spyker dtype string).

        Notes
        -----
        The Python wrapper normalizes `pad` to 4 integers via ``expand4`` and ensures
        `filters` is a sequence before delegating to the C++ constructor.
        """
        p4 = expand4(pad)
        device = impl.device("cpu") if device is None else device
        flist = [filters] if isinstance(filters, DoGFilter) else list(filters)
        super().__init__(device, size, flist, p4, dtype)
        self._pad = p4

    def __call__(self, array: TensorLike) -> TensorLike:
        """
        Apply the DoG filter bank to an input tensor.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Input tensor. Layout is normalized to 4D internally using ``to4``;
            batch and channel handling follow backend conventions.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Output tensor with the same ecosystem (PyTorch/NumPy/Spyker) as `array`,
            allocated with matching dtype and device semantics.

        Notes
        -----
        The function computes the output shape via ``impl.shape.dog`` and allocates
        the result with ``create_array`` to preserve the input ecosystem, then
        calls the C++ ``_forward``.
        """
        input_ = to4(wrap_array(array))
        shape = impl.shape.dog(input_.shape, self.kernel.shape, self._pad)
        output = create_array(array, self.kernel.dtype, shape)
        self._forward(input_, wrap_array(output))
        return output


class Gabor(impl.Gabor):
    r"""
    2D Gabor filtering module.

    Wraps the C++ Gabor operator for orientation- and frequency-selective
    convolution with device/dtype/padding management and cross-ecosystem interop.

    Attributes
    ----------
    kernel : spyker.Tensor
        The assembled convolution kernel bank (read-only from Python).
    device : spyker.Device
        Device where the module's parameters reside (e.g., CPU or CUDA device).
    """

    def __init__(
        self,
        size: int,
        filters: Union[GaborFilter, Sequence[GaborFilter]],
        pad: Union[int, Sequence[int]] = 0,
        device: impl.Device | None = None,
        dtype: DataType = "f32",
    ) -> None:
        """
        Parameters
        ----------
        size : int
            Half window size; the full kernel size is ``2 * size + 1``.
        filters : GaborFilter or sequence of GaborFilter
            One or more Gabor filter parameter sets to instantiate the kernel bank.
        pad : int or sequence of 2 or 4 ints, default=0
            Input padding. If an int, symmetric padding is applied to both axes.
            A sequence of 2 is interpreted as (width, height); a sequence of 4 as
            (left, right, top, bottom).
        device : impl.Device or None, default=None
            Destination device for the module parameters and computations (default CPU).
        dtype : DataType, default="f32"
            Scalar dtype code for parameters/outputs (Spyker dtype string).

        Notes
        -----
        The Python wrapper normalizes `pad` to 4 integers via ``expand4`` and ensures
        `filters` is a sequence before delegating to the C++ constructor.
        """
        p4 = expand4(pad)
        device = impl.device("cpu") if device is None else device
        flist = [filters] if isinstance(filters, GaborFilter) else list(filters)
        super().__init__(device, size, flist, p4, dtype)
        self._pad = p4

    def __call__(self, array: TensorLike) -> TensorLike:
        """
        Apply the Gabor filter bank to an input tensor.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Input tensor. Layout is normalized to 4D internally using ``to4``;
            batch and channel handling follow backend conventions.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Output tensor with the same ecosystem (PyTorch/NumPy/Spyker) as `array`,
            allocated with matching dtype and device semantics.

        Notes
        -----
        The function computes the output shape via ``impl.shape.gabor`` and allocates
        the result with ``create_array`` to preserve the input ecosystem, then
        calls the C++ ``_forward``.
        """
        input_ = to4(wrap_array(array))
        shape = impl.shape.gabor(input_.shape, self.kernel.shape, self._pad)
        output = create_array(array, self.kernel.dtype, shape)
        self._forward(input_, wrap_array(output))
        return output


class LoG(impl.LoG):
    r"""
    2D Laplacian-of-Gaussian (LoG) filtering module.

    Wraps the C++ LoG operator and manages padding, device/dtype placement,
    and zero-copy interop with NumPy/PyTorch/Spyker tensors.

    Attributes
    ----------
    kernel : spyker.Tensor
        Convolution kernel bank (read-only from Python).
    device : spyker.Device
        Device where the module resides (e.g., CPU or CUDA device).
    """

    def __init__(
        self,
        size: int,
        stds: Sequence[float],
        pad: Union[int, Sequence[int]] = 0,
        device: impl.Device | None = None,
        dtype: DataType = "f32",
    ) -> None:
        """
        Parameters
        ----------
        size : int
            Half window size; the full kernel size is ``2 * size + 1``.
        stds : sequence of float
            Standard deviations for the LoG kernels to assemble into the bank.
        pad : int or sequence of 2 or 4 ints, default=0
            Input padding. If an int, symmetric padding is applied. A sequence of 2
            is interpreted as (width, height); a sequence of 4 as (left, right, top, bottom).
        device : impl.Device or None, default=None
            Destination device for parameters and computations (default CPU).
        dtype : DataType, default="f32"
            Scalar dtype code for parameters/outputs (Spyker dtype string).

        Notes
        -----
        The wrapper normalizes `pad` to 4 integers via ``expand4`` before delegating to C++.
        """
        p4 = expand4(pad)
        device = impl.device("cpu") if device is None else device
        super().__init__(device, size, stds, p4, dtype)
        self._pad = p4

    def __call__(self, array: TensorLike) -> TensorLike:
        """
        Apply the LoG filter bank to an input tensor.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Input tensor. Layout is normalized to 4D internally using ``to4``.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Output tensor in the same ecosystem (PyTorch/NumPy/Spyker) as `array`.
        """
        input_ = to4(wrap_array(array))
        shape = impl.shape.log(input_.shape, self.kernel.shape, self._pad)
        output = create_array(array, self.kernel.dtype, shape)
        self._forward(input_, wrap_array(output))
        return output


class ZCA(impl.ZCA):
    r"""
    ZCA Whitening module.

    Learns mean and whitening transform from data and applies the transform in-place
    or out-of-place across PyTorch, NumPy, or Spyker tensors via zero-copy views.

    Attributes
    ----------
    mean : spyker.Tensor
        Learned feature means.
    transform : spyker.Tensor
        Learned whitening matrix.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, array: TensorLike, epsilon: float, transform: bool = False) -> ZCA:
        """
        Fit mean and whitening transform to the provided data.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Input samples. Leading dimensions are interpreted as batch/stack; the last
            dimension typically represents features. Accepted layouts are determined
            by the backend.
        epsilon : float
            Stabilizer added to eigenvalues (or diagonal) to ensure numerical stability.
        transform : bool, default=False
            If True, also transform the input *in-place* during fitting (when supported).

        Returns
        -------
        ZCA
            The fitted instance (for chaining).

        Notes
        -----
        Internally dispatches to the C++ implementation via ``self._fit(...)`` after
        wrapping the input with ``wrap_array`` (zero-copy when possible).
        """
        self._fit(wrap_array(array), epsilon, transform)
        return self

    def __call__(self, *array: TensorLike, inplace: bool = True) -> TensorLike | Tuple[TensorLike, ...]:
        """
        Apply the learned whitening transform to one or more arrays.

        Parameters
        ----------
        *array : torch.Tensor or numpy.ndarray or spyker.Tensor
            One or more inputs to be transformed.
        inplace : bool, default=True
            If True, transform inputs in place when possible. If False, inputs are
            cloned before transformation and the originals are left untouched.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor or tuple
            Transformed object(s) of the same ecosystem as the inputs. Returns a single
            object for one input, or a tuple preserving input order for multiple inputs.

        Notes
        -----
        Each input is wrapped via ``wrap_array`` and passed to ``self._forward(...)``.
        When ``inplace=False``, a deep copy is made using ``clone_array`` prior to
        transformation.
        """
        if not inplace:
            array = [clone_array(x) for x in array]  # type: ignore[assignment]
        [self._forward(wrap_array(x)) for x in array]
        return tuple(array) if len(array) > 1 else array[0]  # type: ignore[return-value]

    @staticmethod
    def split(array: TensorLike) -> TensorLike:
        """
        Split a ZCA-transformed array into two channels: positive and negative.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Input assumed to be ZCA-transformed. The exact required layout is
            backend-dependent.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Output with an added/modified channel dimension holding the positive and
            negative parts, matching the input's ecosystem (PyTorch/NumPy/Spyker).

        Notes
        -----
        This is a thin wrapper over the C++ static method and associated shape helper:
        it computes the target shape via ``impl.shape.zca_split`` and materializes an
        appropriately-typed/placed output using ``create_array`` before invoking
        ``impl.ZCA._split``.
        """
        input_ = least2(wrap_array(array))
        shape = impl.shape.zca_split(input_.shape)
        output = create_array(array, input_.dtype, shape)
        impl.ZCA._split(input_, wrap_array(output))
        return output

    def save(self, path: str) -> None:
        """
        Save the fitted ZCA model to a ``.npz`` file.

        Parameters
        ----------
        path : str
            Output file path. A NumPy ``.npz`` archive is written with keys
            ``mean`` and ``transform``.
        """
        import numpy

        mean, transform = to_numpy(self.mean, self.transform)
        numpy.savez(path, mean=mean, transform=transform)

    def load(self, path: str) -> None:
        """
        Load a fitted ZCA model from a ``.npz`` file.

        Parameters
        ----------
        path : str
            Input file path pointing to a NumPy ``.npz`` archive containing keys
            ``mean`` and ``transform``.
        """
        import numpy

        data = numpy.load(path)
        wrap_array(data["mean"]).to(self.mean)
        wrap_array(data["transform"]).to(self.transform)


class Conv(impl.Conv):
    r"""
    2D convolution module.

    Wraps the C++ convolution operator and manages kernel/stride/padding normalization,
    device/dtype placement, and zero-copy interop with NumPy/PyTorch/Spyker tensors.

    Attributes
    ----------
    kernel : spyker.Tensor
        Convolution kernel bank (read-only from Python).
    device : spyker.Device
        Device where the module resides (e.g., CPU or CUDA device).
    """

    def __init__(
        self,
        insize: int,
        outsize: int,
        kernel: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        pad: Union[int, Sequence[int]] = 0,
        mean: float = 0.5,
        std: float = 0.02,
        device: impl.Device | None = None,
        dtype: DataType = "f32",
    ) -> None:
        """
        Parameters
        ----------
        insize : int
            Number of input channels.
        outsize : int
            Number of output channels.
        kernel : int or sequence of 2 ints
            Convolution kernel size. If an int, expanded to ``(k, k)``.
        stride : int or sequence of 2 ints, default=1
            Convolution stride. If an int, expanded to ``(s, s)``.
        pad : int or sequence of 2 or 4 ints, default=0
            Input padding. If an int, symmetric padding on both axes. A sequence of 2
            is interpreted as (width, height); a sequence of 4 as (left, right, top, bottom).
        mean : float, default=0.5
            Mean of the normal distribution used to initialize the kernel.
        std : float, default=0.02
            Standard deviation of the normal distribution used to initialize the kernel.
        device : spyker.Device or None, default=None
            Destination device for parameters and computations (default CPU).
        dtype : DataType, default="f32"
            Scalar dtype code for parameters/outputs (Spyker dtype string).

        Notes
        -----
        The wrapper normalizes ``kernel``/``stride`` with ``expand2`` and ``pad`` with
        ``expand4`` before delegating to the C++ constructor.
        """
        k2 = expand2(kernel)
        s2 = expand2(stride)
        p4 = expand4(pad)

        device = impl.device("cpu") if device is None else device
        super().__init__(device, insize, outsize, k2, s2, p4, mean, std, dtype)
        self._stride = s2
        self._pad = p4

    def __call__(self, array: TensorLike, threshold: float | None = None) -> TensorLike:
        """
        Apply the convolution to an input tensor.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Input tensor. Layout is normalized to 5D internally using ``to5`` (batch,
            channels, depth=1 if absent, height, width); exact conventions are backend-dependent.
        threshold : float or None, default=None
            Optional sparsification/thresholding control (when supported by the backend).
            If provided, a specialized C++ path is taken.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Convolved output tensor in the same ecosystem (PyTorch/NumPy/Spyker) as `array`.

        Notes
        -----
        When ``threshold`` is None, the wrapper computes the output shape via
        ``impl.shape.conv`` and allocates the destination with ``create_array`` to
        preserve the input ecosystem, then calls the C++ ``_forward``. When a threshold
        is given, the call is forwarded directly to the C++ overload.
        """
        if threshold is not None:
            return self._forward(array, threshold)  # backend handles wrapping/dispatch

        input_ = to5(wrap_array(array))
        shape = impl.shape.conv(input_.shape, self.kernel.shape, self._stride, self._pad)
        output = create_array(array, self.kernel.dtype, shape)
        self._forward(input_, wrap_array(output))
        return output

    def stdp(
        self,
        array: TensorLike,
        winners: Sequence[Sequence[impl.Winner]],
        output: TensorLike | None = None,
    ) -> None:
        """
        Apply STDP weight updates to the convolution layer.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Convolution input tensor provided to the forward pass.
        winners : sequence of sequence of spyker.Winner
            Selected winner neurons per sample/channel for updating.
        output : torch.Tensor or numpy.ndarray or spyker.Tensor or None, default=None
            Convolution output tensor. If omitted, the backend may recompute or use
            internal state as needed.

        Notes
        -----
        If ``output`` is provided, both inputs are wrapped with ``wrap_array`` to
        enable zero-copy views before invoking the C++ ``_stdp``.
        """
        if output is None:
            self._stdp(array, winners)  # backend path that does not require explicit output
        else:
            self._stdp(wrap_array(array), winners, wrap_array(output))


class FC(impl.FC):
    r"""
    Fully connected (dense) module.

    Wraps the C++ dense layer and supports forward, STDP updates, and backpropagation,
    while preserving device/dtype and cross-ecosystem interop.

    Attributes
    ----------
    kernel : spyker.Tensor
        Weight matrix of shape ``(out_features, in_features)`` (backend-specific layout).
    device : spyker.Device
        Device where parameters reside (e.g., CPU or CUDA device).
    """

    def __init__(
        self,
        insize: int,
        outsize: int,
        mean: float = 0.5,
        std: float = 0.02,
        device: impl.Device | None = None,
        dtype: DataType = "f32",
    ) -> None:
        """
        Parameters
        ----------
        insize : int
            Input feature dimension.
        outsize : int
            Output feature dimension.
        mean : float, default=0.5
            Mean of the normal distribution used to initialize the kernel.
        std : float, default=0.02
            Standard deviation of the normal distribution used to initialize the kernel.
        device : impl.Device or None, default=None
            Destination device for parameters and computations (default CPU).
        dtype : DataType, default="f32"
            Scalar dtype code for parameters/outputs (Spyker dtype string).
        """
        device = impl.device("cpu") if device is None else device
        super().__init__(device, insize, outsize, mean, std, dtype)

    def __call__(self, array: TensorLike, sign: bool = False) -> TensorLike:
        """
        Apply the fully connected transform to the input.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Dense input tensor. Layout is normalized to 3D internally with ``to3``.
        sign : bool, default=False
            If True, the forward pass may apply sign-based post-processing (backend-dependent).

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Dense output tensor in the same ecosystem as `array`.
        """
        input_ = to3(wrap_array(array))
        shape = impl.shape.fc(input_.shape, self.kernel.shape)
        output = create_array(array, self.kernel.dtype, shape)
        self._forward(input_, wrap_array(output), sign)
        return output

    def stdp(self, array: TensorLike, winners: Sequence[Sequence[impl.Winner]], output: TensorLike) -> None:
        """
        Apply STDP-based weight updates to the fully connected layer.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Dense input tensor provided to the forward pass.
        winners : sequence of sequence of spyker.Winner
            Selected winner neurons per sample/channel for updating.
        output : torch.Tensor or numpy.ndarray or spyker.Tensor
            Dense output tensor produced by the forward pass.
        """
        self._stdp(wrap_array(array), winners, wrap_array(output))

    def backward(self, array: TensorLike, output: TensorLike, grad: TensorLike) -> TensorLike:
        """
        Backpropagate gradients through the fully connected layer.

        Parameters
        ----------
        array : torch.Tensor or numpy.ndarray or spyker.Tensor
            Dense input tensor provided to the forward pass.
        output : torch.Tensor or numpy.ndarray or spyker.Tensor
            Dense output tensor from the forward pass (used for context if needed).
        grad : torch.Tensor or numpy.ndarray or spyker.Tensor
            Gradient of the loss w.r.t. the output tensor.

        Returns
        -------
        torch.Tensor or numpy.ndarray or spyker.Tensor
            Gradient of the loss w.r.t. the input tensor (`next` gradient), allocated
            in the same ecosystem and with the same shape as the normalized input.
        """
        input_, grad_ = to2(wrap_array(array)), to2(wrap_array(grad))
        next_ = create_array(grad, grad_.dtype, input_.shape)
        self._backward(input_, wrap_array(output), grad_, wrap_array(next_))
        return next_


def canny(array: TensorLike, low: float, high: float) -> TensorLike:
    r"""
    Canny edge detection.

    Wraps the C++ Canny implementation and preserves the input ecosystem
    (PyTorch/NumPy/Spyker) for the returned tensor.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input image or batch. Layout is normalized to 4D internally via ``to4``.
    low : float
        Lower hysteresis threshold for weak edges.
    high : float
        Upper hysteresis threshold for strong edges.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Edge map with dtype ``uint8`` (0/255 semantics), shaped like the normalized input.

    Notes
    -----
    The destination is allocated with dtype ``"u8"`` to match typical Canny outputs.
    """
    input_ = to4(wrap_array(array))
    output = create_array(array, "u8", input_.shape)
    impl.canny(input_, wrap_array(output), low, high)
    return output


def fc(array: TensorLike, kernel: TensorLike, sign: bool = False) -> TensorLike:
    r"""
    Fully connected (dense) operation.

    Computes ``y = W @ x`` over the last dimension (backend-dependent layout),
    preserving the input ecosystem for the output.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input dense tensor. Layout is normalized to 3D internally via ``to3``.
    kernel : torch.Tensor or numpy.ndarray or spyker.Tensor
        Dense weight matrix compatible with the backend's ``impl.shape.fc`` contract.
    sign : bool, default=False
        Run the kernel through the sign function before running the operation.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Output dense tensor in the same ecosystem (PyTorch/NumPy/Spyker) as `array`.

    Notes
    -----
    The output shape is computed with ``impl.shape.fc``; allocation is done via
    ``create_array`` to match dtype/device semantics of the inputs.
    """
    input_ = to3(wrap_array(array))
    kernel_ = wrap_array(kernel)
    shape = impl.shape.fc(input_.shape, kernel_.shape)
    output = create_array(array, kernel_.dtype, shape)
    impl.fc(input_, kernel_, wrap_array(output), sign)
    return output


def conv(
    array: TensorLike,
    kernel: TensorLike,
    stride: Union[int, Sequence[int]] = 1,
    pad: Union[int, Sequence[int]] = 0,
) -> TensorLike:
    r"""
    2D convolution.

    Wraps the C++ convolution operator, normalizing stride/padding and preserving
    the input ecosystem (PyTorch/NumPy/Spyker) for the returned tensor.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Convolution input. Layout is normalized to 5D via ``to5``; exact conventions
        (batch/channels/spatial) are backend-dependent.
    kernel : torch.Tensor or numpy.ndarray or spyker.Tensor
        Convolution kernel tensor. Must be compatible with the backend's expected layout.
    stride : int or sequence of 2 ints, default=1
        Convolution stride. An int is expanded to ``(s, s)``.
    pad : int or sequence of 2 or 4 ints, default=0
        Input padding. An int applies symmetric padding; a sequence of 2 means
        ``(width, height)``, and of 4 means ``(left, right, top, bottom)``.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Convolved output in the same ecosystem as `array`.

    Notes
    -----
    The wrapper expands ``stride`` via ``expand2`` and ``pad`` via ``expand4``,
    computes the output shape with ``impl.shape.conv``, allocates via ``create_array``,
    and then calls the C++ ``impl.conv``.
    """
    kernel_ = wrap_array(kernel)
    s2 = expand2(stride)
    p4 = expand4(pad)

    input_ = to5(wrap_array(array))
    shape = impl.shape.conv(input_.shape, kernel_.shape, s2, p4)
    output = create_array(array, kernel_.dtype, shape)
    impl.conv(input_, kernel_, wrap_array(output), s2, p4)
    return output


def pad(
    array: TensorLike,
    pad: Union[int, Sequence[int]] = 0,
    value: float = 0.0,
) -> TensorLike:
    r"""
    Apply 2D spatial padding.

    Wraps the C++ padding operator and preserves the input ecosystem
    (PyTorch/NumPy/Spyker) for the returned tensor.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor to be padded. Layout is normalized to 5D internally via ``to5``.
    pad : int or sequence of 2 or 4 ints, default=0
        Padding specification:
        - int → symmetric padding on both axes,
        - 2-tuple → (width, height),
        - 4-tuple → (left, right, top, bottom).
    value : float, default=0.0
        Fill value for padded regions.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Padded output tensor in the same ecosystem as `array`.

    Notes
    -----
    The output shape is computed with ``impl.shape.pad``; allocation is done via
    ``create_array`` with the input's dtype/device semantics.
    """
    p4 = expand4(pad)
    input_ = to5(wrap_array(array))
    shape = impl.shape.pad(input_.shape, p4)
    output = create_array(array, input_.dtype, shape)
    impl.pad(input_, wrap_array(output), p4, value)
    return output


def threshold(
    array: TensorLike,
    threshold: float,
    value: float = 0.0,
    inplace: bool = True,
) -> TensorLike:
    r"""
    Apply thresholding.

    All values below the threshold are set to `value`; others remain unchanged.
    Can operate in-place or return a new array.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor.
    threshold : float
        Threshold value.
    value : float, default=0.0
        Replacement value for elements below the threshold.
    inplace : bool, default=True
        If True, modifies `array` directly; if False, a clone is created first.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Thresholded tensor (same object if inplace=True).
    """
    if not inplace:
        array = clone_array(array)
    impl.threshold(wrap_array(array, writeable=True), threshold, value)
    return array


def quantize(
    array: TensorLike,
    lower: float,
    middle: float,
    upper: float,
    inplace: bool = True,
) -> TensorLike:
    r"""
    Quantize values into two levels based on a midpoint.

    If element < `middle`, it is set to `lower`; otherwise set to `upper`.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor to quantize.
    lower : float
        Value to assign when element < `middle`.
    middle : float
        Midpoint threshold for quantization.
    upper : float
        Value to assign when element >= `middle`.
    inplace : bool, default=True
        If True, modifies `array` directly; if False, a clone is created first.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Quantized tensor (same object if inplace=True).
    """
    if not inplace:
        array = clone_array(array)
    impl.quantize(wrap_array(array, writeable=True), lower, middle, upper)
    return array


def code(
    array: TensorLike,
    time: int,
    sort: bool = True,
    dtype: "DataType" = "u8",
    code: CodingType = "rank",
) -> TensorLike:
    r"""
    Spike-time coding.

    Encodes static values into a temporal code across ``time`` steps (e.g., **rank coding**),
    returning an output in the same ecosystem as the input (PyTorch/NumPy/Spyker).

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input dense tensor. Its layout is normalized to at least 2D internally via ``least2``.
    time : int
        Number of time steps for the temporal code.
    sort : bool, default=True
        If True, values may be sorted to improve coding fidelity (at the cost of performance).
    dtype : DataType, default="u8"
        Scalar dtype code for the output tensor.
    code : CodingType, default="rank"
        Coding scheme identifier (e.g., "rank"). Supported values are backend-dependent.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Coded output tensor with shape computed by ``impl.shape.code`` and dtype ``dtype``.

    Notes
    -----
    The output shape is obtained via ``impl.shape.code(input.shape, time)`` and allocated with
    ``create_array`` to preserve the input ecosystem.
    """
    input_ = least2(wrap_array(array))
    shape = impl.shape.code(input_.shape, time)
    output = create_array(array, dtype, shape)
    impl.code(input_, wrap_array(output), time, sort, code)
    return output


def infinite(
    array: TensorLike,
    value: float = 0.0,
    inplace: bool = True,
) -> TensorLike:
    r"""
    Infinite thresholding.

    Marks/sets values considered **non-finite** (implementation-dependent) to ``value``.
    Can modify the input in-place or return a cloned-and-modified copy.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor to sanitize.
    value : float, default=0.0
        Replacement value for elements considered infinite/non-finite.
    inplace : bool, default=True
        If True, modify `array` directly; if False, operate on a deep copy.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        The modified tensor (same object if ``inplace=True``).

    Notes
    -----
    The input is normalized to at least 3D internally via ``least3`` before processing.
    """
    if not inplace:
        array = clone_array(array)
    input_ = least3(wrap_array(array, writeable=True))
    impl.infinite(input_, value)
    return array


def fire(
    array: TensorLike,
    threshold: float = 0.0,
    dtype: DataType = "u8",
    code: CodingType = "rank",
) -> TensorLike:
    r"""
    Integrate-and-fire (IAF) spike generation.

    Converts analog inputs into spike events using an integrate-and-fire rule. If the input
    is already thresholded/quantized, ``threshold`` can be left at its default.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor to convert into spikes.
    threshold : float, default=0.0
        Firing threshold; elements below this may be suppressed (backend-dependent semantics).
    dtype : DataType, default="u8"
        Scalar dtype code for the output spike tensor.
    code : CodingType, default="rank"
        Coding scheme identifier for spike representation (backend-dependent).

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Spike tensor in the same ecosystem as `array`, with the same shape and dtype ``dtype``.
    """
    input_ = wrap_array(array)
    output = create_array(array, dtype, input_.shape)
    impl.fire(input_, wrap_array(output), threshold, code)
    return output


def gather(
    array: TensorLike,
    threshold: float = 0.0,
    dtype: DataType = "u8",
    code: CodingType = "rank",
) -> TensorLike:
    r"""
    Gather temporal information into a compact representation.

    Aggregates events across the time dimension into a single frame (or reduced
    temporal structure) using an integrate-and-fire–like rule with an optional threshold.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor. Its layout is normalized to at least 3D via ``least3``.
    threshold : float, default=0.0
        Firing threshold. If the input is already thresholded/sparse, this can remain 0.
    dtype : DataType, default="u8"
        Scalar dtype code for the output tensor.
    code : CodingType, default="rank"
        Coding scheme identifier (backend-dependent).

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Gathered output tensor in the same ecosystem as `array`. The shape is
        determined by ``impl.shape.gather(input.shape)``.

    Notes
    -----
    Output allocation is performed via ``create_array`` to preserve device/ecosystem.
    """
    input_ = least3(wrap_array(array))
    output = create_array(array, dtype, impl.shape.gather(input_.shape))
    impl.gather(input_, wrap_array(output), threshold, code)
    return output


def scatter(
    array: TensorLike,
    time: int,
    dtype: DataType = "u8",
) -> TensorLike:
    r"""
    Scatter a compact representation across a time axis.

    Expands static inputs into a temporal sequence of length ``time`` according to
    the backend's scattering rule.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor. Normalized to at least 2D via ``least2``.
    time : int
        Number of time steps in the scattered output.
    dtype : DataType, default="u8"
        Scalar dtype code for the output tensor.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Scattered output tensor in the same ecosystem as `array`. Shape is computed with
        ``impl.shape.scatter(input.shape, time)``.
    """
    input_ = least2(wrap_array(array))
    output = create_array(array, dtype, impl.shape.scatter(input_.shape, time))
    impl.scatter(input_, wrap_array(output))
    return output


def pool(
    array: TensorLike,
    kernel: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int], None] = None,
    pad: Union[int, Sequence[int]] = 0,
    rates: TensorLike | None = None,
) -> TensorLike:
    r"""
    2D max pooling.

    Wraps the C++ pooling operator, normalizing kernel/stride/padding and preserving
    the input ecosystem (PyTorch/NumPy/Spyker) for the returned tensor.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Pooling input tensor. Layout is normalized to 5D internally via ``to5``.
    kernel : int or sequence of 2 ints
        Pooling window size. If an int, expanded to ``(k, k)``.
    stride : None or int or sequence of 2 ints, default=None
        Pooling stride. If None, it defaults to ``kernel``. If an int, expanded to ``(s, s)``.
    pad : int or sequence of 2 or 4 ints, default=0
        Spatial padding. If an int, symmetric padding; a sequence of 2 means ``(width, height)``,
        and a sequence of 4 means ``(left, right, top, bottom)``.
    rates : torch.Tensor or numpy.ndarray or spyker.Tensor or None, default=None
        Optional per-location rates/weights (backend-dependent). If omitted, an empty Spyker
        tensor is passed.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Pooled output tensor in the same ecosystem as `array`. Shape computed with
        ``impl.shape.pool(input.shape, kernel, stride, pad)``.

    Notes
    -----
    - ``kernel`` and ``stride`` are expanded via ``expand2``; ``pad`` via ``expand4``.
    - Output allocation is done with ``create_array`` to preserve dtype/device semantics.
    """
    if stride is None:
        stride = kernel
    k2 = expand2(kernel)
    s2 = expand2(stride)
    p4 = expand4(pad)

    input_ = to5(wrap_array(array))
    shape = impl.shape.pool(input_.shape, k2, s2, p4)
    output = create_array(array, input_.dtype, shape)
    rates_ = impl.Tensor() if rates is None else wrap_array(rates)
    impl.pool(input_, wrap_array(output), k2, s2, p4, rates_)
    return output


def inhibit(
    array: TensorLike,
    threshold: float = 0.0,
    inplace: bool = True,
) -> TensorLike:
    r"""
    Lateral inhibition (optionally in-place).

    Suppresses activations within a local neighborhood according to the backend's
    inhibition rule. If the input is already thresholded/sparse, you can keep
    ``threshold=0``.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor to inhibit.
    threshold : float, default=0.0
        Firing threshold used by the backend (applies to floating-point inputs).
    inplace : bool, default=True
        If True, modify `array` directly; if False, a cloned copy is modified.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        The inhibited tensor (same object if ``inplace=True``).
    """
    if not inplace:
        array = clone_array(array)
    impl.inhibit(wrap_array(array, writeable=True), threshold)
    return array


def fcwta(
    array: TensorLike,
    radius: int,
    count: int,
    threshold: float = 0.0,
) -> List[List[impl.Winner]]:
    r"""
    Winner-Take-All (WTA) selection for fully-connected activations.

    Selects `count` winners per local neighborhood (defined by `radius`) using
    the backend's FC WTA rule.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Dense activation tensor (fully-connected layout).
    radius : int
        Neighborhood radius for inhibition/competition.
    count : int
        Number of winners to select per neighborhood.
    threshold : float, default=0.0
        Minimum activation required (used for floating-point inputs).

    Returns
    -------
    list[list[spyker.Winner]]
        Nested winner descriptors for each sample/channel as produced by the backend.
    """
    return impl.fcwta(wrap_array(array), radius, count, threshold)


def convwta(
    array: TensorLike,
    radius: Union[int, Sequence[int]],
    count: int,
    threshold: float = 0.0,
) -> List[List[impl.Winner]]:
    r"""
    Winner-Take-All (WTA) selection for convolutional activations.

    Selects `count` winners per spatial neighborhood using the backend's
    convolutional WTA rule.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Convolutional activation tensor.
    radius : int or sequence of 2 ints
        Neighborhood radius. If an int, expanded to ``(r, r)``.
    count : int
        Number of winners to select per neighborhood.
    threshold : float, default=0.0
        Minimum activation required (used for floating-point inputs).

    Returns
    -------
    list[list[spyker.Winner]]
        Nested winner descriptors for each sample/channel as produced by the backend.
    """
    r2 = expand2(radius)
    return impl.convwta(wrap_array(array), r2, count, threshold)


def stdp(
    conv: impl.Conv,
    array: TensorLike,
    winners: Sequence[Sequence[impl.Winner]],
    output: TensorLike,
) -> None:
    r"""
    Low-level STDP update helper for convolutional layers.

    Dispatches directly to the backend to update `conv`'s parameters using
    the provided winners and forward pass tensors.

    Parameters
    ----------
    conv : impl.Conv
        Convolutional module whose weights will be updated.
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor used for the forward pass.
    winners : sequence of sequence of spyker.Winner
        Selected winner neurons per sample/channel.
    output : torch.Tensor or numpy.ndarray or spyker.Tensor
        Output tensor from the forward pass matching `array`.
    """
    impl.stdp(conv.impl, array, winners, output)


def backward(
    array: TensorLike,
    target: TensorLike,
    time: int,
    gamma: float,
) -> TensorLike:
    r"""
    Temporal backpropagation helper.

    Computes the gradient-like signal w.r.t. the input over a time horizon.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Input tensor from the forward pass.
    target : torch.Tensor or numpy.ndarray or spyker.Tensor
        Target/teaching signal aligned with the forward pass (backend-dependent shape).
    time : int
        Number of time steps to backpropagate through.
    gamma : float
        Temporal discount/decay factor (backend semantics apply).

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        Tensor of the same ecosystem and shape as `array` containing the propagated signal.
    """
    input_ = wrap_array(array)
    output = create_array(array, input_.dtype, input_.shape)
    impl.backward(input_, wrap_array(output), wrap_array(target), time, gamma)
    return output


def labelize(
    array: TensorLike,
    threshold: float,
) -> TensorLike:
    r"""
    Labelize activations into integer class indices.

    Collapses per-sample activations to a single integer label based on the
    backend's selection rule and the given threshold.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Activation tensor; first dimension is interpreted as batch.
    threshold : float
        Minimum activation required to assign a label.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        1D tensor of length ``batch_size`` with dtype ``int64`` (Spyker ``"i64"``),
        returned in the same ecosystem as `array`.
    """
    input_ = wrap_array(array)
    output = create_array(array, "i64", [input_.shape[0]])
    impl.labelize(input_, wrap_array(output), threshold)
    return output
