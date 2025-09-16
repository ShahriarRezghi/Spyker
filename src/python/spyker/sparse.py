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

from typing import List, Sequence, Union

import spyker.spyker_plugin as impl
from spyker.private import expand2, expand4, to4
from spyker.utils import DataType, TensorLike, wrap_array


def code(
    array: TensorLike,
    time: int,
    sort: bool = True,
) -> impl.SparseTensor:
    r"""
    Sparse spike-time coding.

    Encodes a dense input into a **sparse temporal representation** across ``time`` steps
    (e.g., rank coding) and returns a Spyker sparse tensor.

    Parameters
    ----------
    array : torch.Tensor or numpy.ndarray or impl.Tensor
        Dense input tensor. Layout is normalized to 4D internally via ``to4``.
    time : int
        Number of time steps for the temporal code.
    sort : bool, default=True
        If True, values may be sorted to improve coding fidelity (at the cost of performance).

    Returns
    -------
    impl.SparseTensor
        Sparse temporal representation produced by the backend.
    """
    input_ = to4(wrap_array(array))
    return impl.sparse.code(input_, time, sort)


def conv(
    array: impl.SparseTensor,
    kernel: TensorLike,
    threshold: float,
    stride: Union[int, Sequence[int]] = 1,
    pad: Union[int, Sequence[int]] = 0,
) -> impl.SparseTensor:
    r"""
    Sparse 2D convolution.

    Convolves a **sparse** input with a **dense** kernel using the backend's sparse
    convolution, with optional thresholding and normalized stride/padding.

    Parameters
    ----------
    array : impl.SparseTensor
        Sparse input activations.
    kernel : torch.Tensor or numpy.ndarray or impl.Tensor
        Dense convolution kernel bank.
    threshold : float
        Post-convolution threshold used by the backend to control sparsity.
    stride : int or sequence of 2 ints, default=1
        Convolution stride. An int is expanded to ``(s, s)``.
    pad : int or sequence of 2 or 4 ints, default=0
        Spatial padding. An int applies symmetric padding; a sequence of 2 means
        ``(width, height)``, and a sequence of 4 means ``(left, right, top, bottom)``.

    Returns
    -------
    impl.SparseTensor
        Sparse convolution output.
    """
    s2, p4 = expand2(stride), expand4(pad)
    return impl.sparse.conv(array, kernel, threshold, s2, p4)


def gather(
    array: impl.SparseTensor,
    dtype: DataType = "u8",
) -> impl.Tensor:
    r"""
    Gather sparse temporal information into a dense frame.

    Aggregates events across the time dimension of a sparse tensor into a **dense**
    Spyker tensor with the requested dtype.

    Parameters
    ----------
    array : impl.SparseTensor
        Sparse temporal input.
    dtype : DataType, default="u8"
        Scalar dtype code for the dense output tensor.

    Returns
    -------
    impl.Tensor
        Dense gathered output produced by the backend.
    """
    return impl.sparse.gather(array, dtype)


def pool(
    array: impl.SparseTensor,
    kernel: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int], None] = None,
    pad: Union[int, Sequence[int]] = 0,
) -> impl.SparseTensor:
    r"""
    Sparse 2D max pooling.

    Applies max pooling to a sparse activation map, normalizing kernel/stride/padding
    for the backend.

    Parameters
    ----------
    array : impl.SparseTensor
        Sparse input activations.
    kernel : int or sequence of 2 ints
        Pooling window size. If an int, expanded to ``(k, k)``.
    stride : None or int or sequence of 2 ints, default=None
        Pooling stride. If None, it defaults to ``kernel``. If an int, expanded to ``(s, s)``.
    pad : int or sequence of 2 or 4 ints, default=0
        Spatial padding. An int applies symmetric padding; a sequence of 2 means
        ``(width, height)``, and a sequence of 4 means ``(left, right, top, bottom)``.

    Returns
    -------
    impl.SparseTensor
        Pooled sparse output.
    """
    if stride is None:
        stride = kernel
    k2 = expand2(kernel)
    s2 = expand2(stride)
    p4 = expand4(pad)
    return impl.sparse.pool(array, k2, s2, p4)


def inhibit(array: impl.SparseTensor) -> impl.SparseTensor:
    r"""
    Lateral inhibition on sparse activations.

    Suppresses events within local neighborhoods according to the backend's sparse
    inhibition rule.

    Parameters
    ----------
    array : impl.SparseTensor
        Sparse input activations.

    Returns
    -------
    impl.SparseTensor
        Inhibited sparse output.
    """
    return impl.sparse.inhibit(array)


def convwta(
    array: impl.SparseTensor,
    radius: Union[int, Sequence[int]],
    count: int,
) -> List[List[impl.Winner]]:
    r"""
    Winner-Take-All (WTA) selection on sparse convolutional activations.

    Selects `count` winners per spatial neighborhood defined by `radius`.

    Parameters
    ----------
    array : impl.SparseTensor
        Sparse convolutional activations.
    radius : int or sequence of 2 ints
        Neighborhood radius. If an int, expanded to ``(r, r)``.
    count : int
        Number of winners to select per neighborhood.

    Returns
    -------
    list[list[impl.Winner]]
        Nested winner descriptors for each sample/channel as produced by the backend.
    """
    r2 = expand2(radius)
    return impl.sparse.convwta(array, r2, count)
