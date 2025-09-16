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

from typing import Sequence, Tuple

from spyker.spyker_plugin import Tensor


def least2(array: Tensor) -> Tensor:
    shape = list(array.shape)
    if len(shape) == 1:
        shape.insert(0, 1)
    if len(shape) <= 1:
        raise ValueError("Input dimensions couldn't be viewed as at least 2D.")
    return array.reshape(shape)


def least3(array: Tensor) -> Tensor:
    shape = list(array.shape)
    if len(shape) == 2:
        shape.insert(0, 1)
    if len(shape) <= 2:
        raise ValueError("Input dimensions couldn't be viewed as at least 3D.")
    return array.reshape(shape)


def to2(array: Tensor) -> Tensor:
    shape = list(array.shape)
    if len(shape) == 1:
        shape.insert(0, 1)
    if len(shape) != 2:
        raise ValueError("Input dimensions couldn't be viewed as 2D.")
    return array.reshape(shape)


def to3(array: Tensor) -> Tensor:
    shape = list(array.shape)
    if len(shape) == 2:
        shape.insert(0, 1)
    if len(shape) != 3:
        raise ValueError("Input dimensions couldn't be viewed as 3D.")
    return array.reshape(shape)


def to4(array: Tensor) -> Tensor:
    shape = list(array.shape)
    if len(shape) == 3:
        shape.insert(0, 1)
    if len(shape) != 4:
        raise ValueError("Input dimensions couldn't be viewed as 4D.")
    return array.reshape(shape)


def to5(array: Tensor) -> Tensor:
    shape = list(array.shape)
    if len(shape) == 4:
        shape.insert(0, 1)
    if len(shape) != 5:
        raise ValueError("Input dimensions couldn't be viewed as 5D.")
    return array.reshape(shape)


def expand2(shape: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(shape, int):
        return (shape, shape)
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        return shape[0], shape[0]
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return shape[0], shape[1]
    raise ValueError("Given shape couldn't be expanded to 2D.")


def expand4(shape: int | Sequence[int]) -> Tuple[int, int, int, int]:
    if isinstance(shape, int):
        return (shape, shape, shape, shape)
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        return shape[0], shape[0], shape[0], shape[0]
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return shape[0], shape[1], shape[0], shape[1]
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        return shape[0], shape[1], shape[2], shape[3]
    raise ValueError("Given shape couldn't be expanded to 4D.")
