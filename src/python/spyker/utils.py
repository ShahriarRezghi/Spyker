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

from typing import Literal, Sequence, Tuple

import spyker.spyker_plugin as impl

torch_available = True
try:
    import torch
except Exception:
    torch_available = False


numpy_available = True
try:
    import numpy
except Exception:
    numpy_available = False


CodingType = Literal["rank", "rate"]
TensorLike = "torch.Tensor | numpy.ndarray | impl.Tensor"
DataType = Literal["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64"]


def _from_torch_type(dtype: torch.dtype) -> DataType:
    if dtype == torch.int8:
        return "i8"
    if dtype == torch.int16:
        return "i16"
    if dtype == torch.int32:
        return "i32"
    if dtype == torch.int64:
        return "i64"
    if dtype == torch.uint8:
        return "u8"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float64:
        return "f64"
    raise TypeError(f"Given PyTorch tensor data type {dtype} is not supported.")


def _from_torch_device(device: torch.device) -> impl.device:
    if device.type == "cpu":
        return impl.device("cpu")
    if device.type == "cuda":
        return impl.device("cuda", device.index)
    raise TypeError(f"Given PyTorch tensor device {device} is not supported.")


def _to_torch_type(dtype: DataType) -> torch.dtype:
    if dtype == "i8":
        return torch.int8
    if dtype == "i16":
        return torch.int16
    if dtype == "i32":
        return torch.int32
    if dtype == "i64":
        return torch.int64
    if dtype == "u8":
        return torch.uint8
    if dtype == "f16":
        return torch.float16
    if dtype == "f32":
        return torch.float32
    if dtype == "f64":
        return torch.float64
    raise TypeError(f"Given data data type {dtype} is not supported by PyTorch.")


def _to_torch_device(device: impl.device) -> torch.device:
    if device.kind == "cpu":
        return torch.device("cpu")
    if device.kind == "cuda":
        return torch.device(f"cuda:{device.index}")
    raise TypeError(f"Given device {device} is not supported by PyTorch.")


def _from_numpy_type(dtype: type) -> DataType:
    if dtype == numpy.int8:
        return "i8"
    if dtype == numpy.int16:
        return "i16"
    if dtype == numpy.int32:
        return "i32"
    if dtype == numpy.int64:
        return "i64"
    if dtype == numpy.uint8:
        return "u8"
    if dtype == numpy.uint16:
        return "u16"
    if dtype == numpy.uint32:
        return "u32"
    if dtype == numpy.uint64:
        return "u64"
    if dtype == numpy.float16:
        return "f16"
    if dtype == numpy.float32:
        return "f32"
    if dtype == numpy.float64:
        return "f64"
    raise TypeError("Given Numpy array data type {dtype} is not supported.")


def _to_numpy_type(dtype: DataType) -> type:
    if dtype == "i8":
        return numpy.int8
    if dtype == "i16":
        return numpy.int16
    if dtype == "i32":
        return numpy.int32
    if dtype == "i64":
        return numpy.int64
    if dtype == "u8":
        return numpy.uint8
    if dtype == "u16":
        return numpy.uint16
    if dtype == "u32":
        return numpy.uint32
    if dtype == "u64":
        return numpy.uint64
    if dtype == "f16":
        return numpy.float16
    if dtype == "f32":
        return numpy.float32
    if dtype == "f64":
        return numpy.float64
    raise TypeError(f"Given data type {dtype} is not supported by Numpy.")


def _wrap_torch_tensor(array: torch.Tensor) -> impl.Tensor:
    if not array.is_contiguous():
        raise TypeError('Input array is not contiguous. use "contiguous" function to make it contiguous.')
    return create_tensor(
        _from_torch_device(array.device), _from_torch_type(array.dtype), array.shape, data=array.data_ptr()
    )


def _wrap_numpy_array(array: numpy.ndarray, writeable: bool) -> impl.Tensor:
    if not array.flags["C_CONTIGUOUS"]:
        raise TypeError('Input array is not contiguous. use "numpy.ascontiguousarray" function to make it contiguous.')

    if writeable and not array.flags["WRITEABLE"]:
        raise TypeError('Input array is not writable. use "numpy.array(..., copy=True)" to make it writable.')

    ptr, _ = array.__array_interface__["data"]
    return create_tensor(impl.device("cpu"), _from_numpy_type(array.dtype), array.shape, data=ptr)


def _create_torch_tensor(array: torch.Tensor, dtype: DataType, shape: Sequence[int]) -> torch.Tensor:
    return torch.zeros(shape, dtype=_to_torch_type(dtype), device=array.device)


def _create_numpy_array(array: numpy.ndarray, dtype: DataType, shape: Sequence[int]) -> numpy.ndarray:
    return numpy.zeros(shape, dtype=_to_numpy_type(dtype))


def create_tensor(
    device: impl.Device,
    dtype: DataType,
    shape: Sequence[int],
    pinned: bool = False,
    unified: bool = False,
    data: int | None = None,
) -> impl.Tensor:
    """
    Create a Spyker tensor on the given device.

    Parameters
    ----------
    device : spyker.Device
        Target device (e.g., CPU or CUDA with optional index) returned by Spyker's device factory.
    dtype : {"i8","i16","i32","i64","u8","u16","u32","u64","f16","f32","f64"}
        Spyker scalar data type code.
    shape : Sequence[int]
        Tensor shape (row-major). Must be non-negative; empty sequence creates a scalar.
    pinned : bool, default=False
        If True and `device` is CPU, allocate page-locked (pinned) host memory to speed up
        host↔device transfers.
    unified : bool, default=False
        If True and supported, allocate CUDA Unified Memory to allow the same pointer to be
        accessed by host and device. Ignored if not applicable.
    data : int or None, default=None
        Optional raw pointer (as an integer) to pre-allocated memory. If provided, Spyker will
        wrap the buffer instead of allocating new memory. The caller is responsible for lifetime
        guarantees when wrapping external memory.

    Returns
    -------
    spyker.Tensor
        Newly created Spyker tensor view/owner consistent with the provided arguments.

    Raises
    ------
    ValueError
        If `shape` contains negative dimensions.
    TypeError
        If arguments are incompatible with the target device or data type.

    Notes
    -----
    - When `data` is None, memory is allocated by Spyker using `pinned`/`unified` hints.
    - When `data` is not None, Spyker does not copy; it constructs a tensor wrapping the given
      pointer with the specified dtype/shape/device.
    """

    if data is None:
        return impl.tensor(device, dtype, shape, pinned, unified)
    else:
        return impl.tensor(data, device, dtype, shape, pinned, unified)


def wrap_array(array: TensorLike, writeable: bool = False) -> impl.Tensor:
    """
    Create a zero-copy Spyker tensor view over a PyTorch tensor, NumPy array, or return the
    same object if it is already a Spyker tensor.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Source array to wrap. Requires PyTorch/NumPy to be installed if those types are used.
    writeable : bool, default=False
        If the result should be writable and the array is a non-writeable NumPy array, an error
        will be raised.

    Returns
    -------
    spyker.Tensor
        Spyker tensor view referencing the same memory (no data copy).

    Raises
    ------
    TypeError
        If `array` is not a supported type or if the corresponding optional dependency is
        not available.
    TypeError
        If writeable is true and `array` is a read-only NumPy array

    Notes
    -----
    - This function avoids data copies; changes through either view reflect in the other,
      subject to backend semantics.
    - If `array` is a NumPy array with `writeable=False` and `writeable=True` is requested,
      an error may be raised
    """

    if torch_available and torch.is_tensor(array):
        return _wrap_torch_tensor(array)

    if numpy_available and type(array) is numpy.ndarray:
        return _wrap_numpy_array(array, writeable=writeable)

    if type(array) is impl.Tensor:
        return array

    raise TypeError(
        f"Input array {type(array)} can only be Numpy array or PyTorch tensor (if installed) or Spyker tensor."
    )


def create_array(array: TensorLike, dtype: torch.dtype | type | DataType, shape: Sequence[int]) -> TensorLike:
    """
    Allocate a new array of the *same high-level kind* as `array` with the given dtype and shape.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Reference object whose kind determines the return type (PyTorch, NumPy, or Spyker).
    dtype : torch.dtype or numpy.dtype/type or Spyker DataType
        Target scalar type. Accepts:
        - `torch.dtype` for PyTorch outputs,
        - `numpy.dtype`/Python type for NumPy outputs,
        - Spyker dtype code for Spyker outputs.
    shape : Sequence[int]
        Desired shape of the new array.

    Returns
    -------
    torch.Tensor or numpy.ndarray or impl.Tensor
        Newly allocated array/tensor consistent with the input ecosystem.

    Raises
    ------
    TypeError
        If `array` is of an unsupported kind or the required optional dependency is not available.
    ValueError
        If `shape` contains invalid dimensions.

    Examples
    --------
    >>> # If `array` is a torch.Tensor, returns a torch.Tensor on the same device
    >>> # If `array` is a numpy.ndarray, returns a NumPy array
    >>> # If `array` is a Spyker tensor, returns a Spyker tensor
    """

    if torch_available and torch.is_tensor(array):
        return _create_torch_tensor(array, dtype, shape)

    if numpy_available and type(array) is numpy.ndarray:
        return _create_numpy_array(array, dtype, shape)

    if type(array) is impl.Tensor:
        return create_tensor(array.device, dtype, shape)

    raise TypeError(
        f"Input array {type(array)} can only be Numpy array or PyTorch tensor (if installed) or Spyker tensor."
    )


def clone_array(array: TensorLike) -> TensorLike:
    """
    Return a deep copy of `array` preserving its ecosystem (PyTorch, NumPy, or Spyker).

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Source array to clone.

    Returns
    -------
    torch.Tensor or numpy.ndarray or spyker.Tensor
        A new object with identical contents and independent storage.

    Notes
    -----
    - For PyTorch tensors, this calls ``tensor.clone()``.
    - For NumPy arrays, this returns ``numpy.array(array, copy=True)``.
    - For Spyker tensors, this uses ``spyker.Tensor.copy()``.
    """

    if torch_available and torch.is_tensor(array):
        return array.clone()

    if numpy_available and type(array) is numpy.ndarray:
        return numpy.array(array, copy=True)

    if type(array) is impl.Tensor:
        return array.copy()


def copy_array(source: TensorLike, destin: TensorLike) -> None:
    """
    Copy data from `source` into `destin`, converting/dispatching through Spyker views.

    Parameters
    ----------
    source : torch.Tensor or numpy.ndarray or spyker.Tensor
        Data source.
    destin : torch.Tensor or numpy.ndarray or spyker.Tensor
        Destination buffer. Shape and dtype must be compatible with `source` after any
        implicit conversions.

    Raises
    ------
    TypeError
        If either input type is unsupported or required dependencies are missing.
    ValueError
        If shapes/dtypes are incompatible for copy.

    Notes
    -----
    This function wraps both arrays as Spyker tensors (zero-copy views when possible) and
    performs ``source_view.to(dest_view)`` for efficient cross-ecosystem transfers.
    """

    wrap_array(source, False).to(wrap_array(destin, True))


def _to_tensor(array: TensorLike, pinned: bool = False, unified: bool = False) -> impl.Tensor:
    temp = wrap_array(array)
    output = create_tensor(temp.device, temp.dtype, temp.shape, pinned, unified)
    temp.to(output)
    return output


def _to_numpy(array: impl.Tensor) -> numpy.ndarray:
    dtype = _to_numpy_type(array.dtype)
    output = numpy.zeros(array.shape, dtype=dtype)
    array.to(wrap_array(output))
    return output


def _to_torch(array: impl.Tensor) -> torch.Tensor:
    dtype = _to_torch_type(array.dtype)
    device = _to_torch_device(array.device)
    output = torch.zeros(array.shape, dtype=dtype, device=device)
    array.to(wrap_array(output))
    return output


def _to_sparse(array: TensorLike, threshold: float = 0.0) -> impl.SparseTensor:
    return impl.sparse_tensor(wrap_array(array), threshold)


def to_tensor(
    *arrays: TensorLike, pinned: bool = False, unified: bool = False
) -> impl.Tensor | tuple[impl.Tensor, ...]:
    """
    Convert one or more arrays (PyTorch/NumPy/Spyker) to new Spyker tensors.

    Parameters
    ----------
    arrays : torch.Tensor or numpy.ndarray or spyker.Tensor
        One or more input arrays to convert.
    pinned : bool, default=False
        Use pinned host memory for destinations where applicable.
    unified : bool, default=False
        Use CUDA Unified Memory for destinations where applicable.

    Returns
    -------
    spyker.Tensor or tuple of spyker.Tensor
        A single Spyker tensor if one input is given; otherwise, a tuple in the same order.

    Examples
    --------
    >>> x_spyker = to_tensor(x_numpy)
    >>> a_spyker, b_spyker = to_tensor(a_torch, b_numpy, pinned=True)
    """

    output = tuple([_to_tensor(x, pinned, unified) for x in arrays])
    return output if len(output) > 1 else output[0]


def to_numpy(*arrays: impl.Tensor) -> numpy.ndarray:
    """
    Convert one or more Spyker tensors to NumPy arrays.

    Parameters
    ----------
    arrays : spyker.Tensor
        One or more Spyker tensors.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        A single NumPy array if one input is given; otherwise, a tuple in the same order.

    Raises
    ------
    TypeError
        If NumPy is not available.
    """

    output = tuple([_to_numpy(x) for x in arrays])
    return output if len(output) > 1 else output[0]


def to_torch(*arrays: impl.Tensor) -> torch.Tensor:
    """
    Convert one or more Spyker tensors to PyTorch tensors.

    Parameters
    ----------
    arrays : spyker.Tensor
        One or more Spyker tensors.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        A single PyTorch tensor if one input is given; otherwise, a tuple in the same order.

    Raises
    ------
    TypeError
        If PyTorch is not available.
    """

    output = tuple([_to_torch(x) for x in arrays])
    return output if len(output) > 1 else output[0]


def to_sparse(*arrays: TensorLike, threshold: float = 0.0) -> impl.SparseTensor:
    """
    Convert one or more dense arrays to Spyker sparse tensors.

    Parameters
    ----------
    arrays : torch.Tensor or numpy.ndarray or spyker.Tensor
        One or more dense inputs to be sparsified.
    threshold : float, default=0.0
        Pruning threshold

    Returns
    -------
    spyker.SparseTensor or tuple of spyker.SparseTensor
        A single sparse tensor if one input is given; otherwise, a tuple in the same order.

    Notes
    -----
    Each input is wrapped as a Spyker view with ``wrap_array`` and passed to
    ``spyker.sparse_tensor`` with the provided `threshold`.
    """

    output = tuple([_to_sparse(x, threshold) for x in arrays])
    return output if len(output) > 1 else output[0]


def read_mnist(
    train_images: str, train_labels: str, test_images: str, test_labels: str
) -> Tuple[impl.Tensor, impl.Tensor, impl.Tensor, impl.Tensor]:
    """
    Load MNIST dataset files into Spyker tensors.

    Parameters
    ----------
    train_images : str
        Path to training images (e.g., ``train-images-idx3-ubyte`` or gzipped counterpart).
    train_labels : str
        Path to training labels (e.g., ``train-labels-idx1-ubyte`` or gzipped counterpart).
    test_images : str
        Path to test images.
    test_labels : str
        Path to test labels.

    Returns
    -------
    (TRX, TRY, TEX, TEY) : tuple of spyker.Tensor
        Training images, training labels, test images, test labels as Spyker tensors.
    """

    TRX = impl.helper.mnist_data(train_images)
    TRY = impl.helper.mnist_label(train_labels)
    TEX = impl.helper.mnist_data(test_images)
    TEY = impl.helper.mnist_label(test_labels)
    return TRX, TRY, TEX, TEY


def read_image(
    path: str,
    mode: Literal["", "Y", "YA", "RGB", "RGBA"] = "",
    size: Tuple[int, int] = (-1, -1),
) -> impl.Tensor:
    """
    Read an image from disk into a Spyker tensor.

    Parameters
    ----------
    path : str
        Path to the image file to read.
    mode : {"" , "Y", "YA", "RGB", "RGBA"}, default=""
        Channel selection:
        - "" → keep the file's original channels
        - "Y" → single-channel luminance
        - "YA" → luminance + alpha
        - "RGB" → 3-channel color
        - "RGBA" → 4-channel color with alpha
    size : (int, int), default=(-1, -1)
        Target (width, height). Use (-1, -1) to keep the original size.

    Returns
    -------
    spyker.Tensor
        Image tensor. On failure to read, an **empty tensor** is returned.

    Notes
    -----
    This wraps the C++ API:
    ``Helper::readImage(path, mode="", size={-1, -1})``.
    """
    return impl.helper.read_image(path, mode, size)


def write_image(
    array: ArrayLike,
    path: str,
    format: Literal["PNG", "BMP", "TGA", "JPG"] = "PNG",
) -> bool:
    """
    Write an image tensor/array to disk.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or spyker.Tensor
        Image data to write. Shape/layout and dtype must be compatible with the
        selected format and backend conversion (e.g., uint8 for 8-bit images).
    path : str
        Destination file path (extension is not required but recommended).
    format : {"PNG", "BMP", "TGA", "JPG"}, default="PNG"
        Output image format.

    Returns
    -------
    bool
        True on success, False on failure.

    Notes
    -----
    This wraps the C++ API:
    ``Helper::writeImage(input, path, format)``.
    The array is first wrapped via ``wrap_array`` for zero-copy interop when possible.
    """
    return impl.helper.write_image(wrap_array(array), path, format)


def read_csv(path: str, delim: str = ", ") -> Tuple[bool, Sequence[Sequence[str]]]:
    """
    Read a delimited text file (CSV/TSV-like) into rows of strings.

    Parameters
    ----------
    path : str
        Path to the CSV (or CSV-like) file.
    delim : str, default=", "
        Field delimiter used to split columns. Defaults to comma + space (", ").

    Returns
    -------
    (ok, rows) : tuple[bool, Sequence[Sequence[str]]]
        - ok: True if the file was read successfully, False otherwise
        - rows: parsed rows as sequences of string fields (empty on failure)

    Notes
    -----
    Thin wrapper over ``Helper::CSV``.
    """
    return impl.helper.read_csv(path, delim)
