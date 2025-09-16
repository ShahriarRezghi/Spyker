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

import spyker.sparse
from spyker.module import (
    FC,
    ZCA,
    BPConfig,
    Conv,
    DoG,
    DoGFilter,
    Gabor,
    GaborFilter,
    LoG,
    STDPConfig,
    backward,
    canny,
    code,
    conv,
    convwta,
    fc,
    fcwta,
    fire,
    gather,
    infinite,
    inhibit,
    labelize,
    pad,
    pool,
    quantize,
    scatter,
    stdp,
    threshold,
)
from spyker.spyker_plugin import SparseTensor, Tensor, Winner, device, version
from spyker.spyker_plugin.control import (
    all_devices,
    clear_context,
    cuda_arch_list,
    cuda_available,
    cuda_cache_clear,
    cuda_cache_enable,
    cuda_cache_enabled,
    cuda_cache_print,
    cuda_conv_force,
    cuda_conv_heuristic,
    cuda_conv_light,
    cuda_current_device,
    cuda_device_arch,
    cuda_device_count,
    cuda_memory_free,
    cuda_memory_taken,
    cuda_memory_total,
    cuda_memory_used,
    cuda_set_device,
    max_threads,
    random_seed,
    set_max_threads,
)
from spyker.utils import copy_array, create_tensor, read_mnist, to_numpy, to_tensor, to_torch, wrap_array
