// BSD 3-Clause License
//
// Copyright (c) 2022-2025, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "impl.h"

#include <ctime>

namespace Spyker
{
std::mt19937 Generator(time(nullptr));

Timer TTT;

namespace Core
{
void random_seed(Size seed) { Generator = std::mt19937(seed); }

#ifndef SPYKER_USE_CUDA
bool cuda_available() { return false; }

Size cuda_device_count() { return 0; }

void cuda_set_device(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_current_device() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

std::vector<Size> cuda_arch_list() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

std::vector<Size> cuda_device_arch() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_total() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_free() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_taken(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_used(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_sync() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

bool cuda_cache_enabled(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_cache_enable(bool enabled, Size device)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void cuda_cache_clear(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_cache_print(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_conv_clear() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_poisson_clear() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_conv_options(Size light, Size heuristic, Size force)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void *pinned_alloc(Size size) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void pinned_dealloc(void *data) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void *unified_alloc(Size size) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void unified_dealloc(void *data) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_copy(Size size, void *input, void *output)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void cpu2cuda(Size size, void *input, void *output)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void cuda2cpu(Size size, void *input, void *output)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}
#endif

CudaSwitch::CudaSwitch(Device device)
{
    enable = (device == Kind::CUDA);
    if (!enable) return;
    previous = cuda_current_device();
    cuda_set_device(device.index());
}

CudaSwitch::~CudaSwitch()
{
    if (enable) cuda_set_device(previous);
}

void transfer(Size size, void *input, Device source, void *output, Device destin)
{
    CudaSwitch sw(source != Kind::CUDA ? destin : source);
    if (source == Kind::CPU && destin == Kind::CUDA)
        cpu2cuda(size, input, output);
    else if (source == Kind::CUDA && destin == Kind::CPU)
        cuda2cpu(size, input, output);
    else if (source == Kind::CUDA && destin == Kind::CUDA)
        cuda_copy(size, input, output);
}

bool compatible(const std::vector<Device> &devices)
{
    if (devices.empty()) return false;
    auto kind = devices.front().kind();
    auto index = devices.front().index();

    for (auto device : devices)
    {
        if (device.kind() != kind) return false;
        if (device.index() != index && device != Kind::CUDA) return false;
    }
    return true;
}

std::vector<Device> devices()
{
    std::vector<Device> devices = {Device(Kind::CPU)};
    Size count = cuda_device_count();
    for (Size i = 0; i < count; ++i) devices.push_back(Device(Kind::CUDA, i));
    return devices;
}
}  // namespace Core
}  // namespace Spyker
