// BSD 3-Clause License
//
// Copyright (c) 2022, Shahriar Rezghi <shahriar25.ss@gmail.com>,
//                     Mohammad-Reza A. Dehaqani <dehaqani@ut.ac.ir>,
//                     University of Tehran
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

#pragma once

#include <spyker/base.h>
#include <spyker/utils.h>

#include <vector>

#define I ,

#define Declare(sign, name, __VA__ARGS__) \
    sign cpu_##name(__VA__ARGS__);        \
    sign cuda_##name(__VA__ARGS__);       \
    inline sign name(__VA__ARGS__, Device device)

#ifdef SPYKER_USE_CUDA
#define Choose(func, ...)                \
    if (device == Kind::CPU)             \
        return cpu_##func(__VA_ARGS__);  \
    else if (device == Kind::CUDA)       \
        return cuda_##func(__VA_ARGS__); \
    else                                 \
        SpykerAssert(false, "Core", "Given device is not recognized.")
#else
#define Choose(func, ...)                                                 \
    if (device == Kind::CPU)                                              \
        return cpu_##func(__VA_ARGS__);                                   \
    else if (device == Kind::CUDA)                                        \
    {                                                                     \
        SpykerAssert(false, "Core", "CUDA is not enabled in this build.") \
    }                                                                     \
    else                                                                  \
        SpykerAssert(false, "Core", "Given device is not recognized.")
#endif

// TODO
// peer to peer should be enabled when copying from device to device or using memory on other devices in cuda functions.

namespace Spyker
{
namespace Core
{
void random_seed(Size seed);

bool cuda_available();

Size cuda_device_count();

void cuda_set_device(Size device);

Size cuda_current_device();

float cuda_current_arch();

std::vector<Size> cuda_arch_list();

std::vector<Size> cuda_device_arch();

Size cuda_memory_total();

Size cuda_memory_free();

Size cuda_memory_taken(Size device);

Size cuda_memory_used(Size device);

bool cuda_cache_enabled(Size device);

void cuda_cache_enable(bool enabled, Size device);

void cuda_cache_clear(Size device);

void cuda_cache_print(Size device);

void *pinned_alloc(Size size);

void pinned_dealloc(void *data);

void *unified_alloc(Size size);

void unified_dealloc(void *data);

void cpu2cuda(Size size, void *input, void *output);

void cuda2cpu(Size size, void *input, void *output);

void cpu_conv_clear();

void cpu_fc_clear();

void cpu_poisson_clear();

void cuda_conv_clear();

void cuda_poisson_clear();

void cuda_conv_options(Size light, Size heuristic, Size force);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct CudaSwitch
{
    Size previous = -1;
    bool enable = false;
    CudaSwitch(Device device);
    ~CudaSwitch();
};

Declare(void *, alloc, Size size)
{
    CudaSwitch sw(device);
    Choose(alloc, size);
}

Declare(void, dealloc, void *data)
{
    CudaSwitch sw(device);
    Choose(dealloc, data);
}

Declare(void, copy, Size size I void *input I void *output)
{
    CudaSwitch sw(device);
    Choose(copy, size, input, output);
}

Declare(void, cast, Size size I Dyn input I Dyn output)
{
    CudaSwitch sw(device);
    Choose(cast, size, input, output);
}

Declare(void, fill, Size size I Dyn input I Scalar value)
{
    CudaSwitch sw(device);
    Choose(fill, size, input, value);
}

void transfer(Size size, void *input, Device source, void *output, Device destin);

bool compatible(const std::vector<Device> &devices);

std::vector<Device> devices();

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void normal_kernel(Vec1<F64> kernel, F64 mean, F64 std);

void gaussian_kernel(Vec2<F64> kernel, F64 std);

void gabor_kernel(Vec2<F64> kernel, F64 sigma, F64 theta, F64 gamma, F64 lambda, F64 psi);

void log_kernel(Vec2<F64> kernel, F64 std);

Vec2<U16> poisson_create(Size time, Size bins);

void chw2hwc(Len3 chw, U8 *input, U8 *output);

void hwc2chw(Len3 chw, U8 *input, U8 *output);

Declare(void, canny, Dyn3 input I Dyn3 output I Scalar low I Scalar high)  //
{
    Choose(canny, input, output, low, high);
}
Declare(void, dog, Dyn4 input I Dyn4 kernel I Dyn4 output I Len4 pad)  //
{
    Choose(dog, input, kernel, output, pad);
}
Declare(void, gabor, Dyn4 input I Dyn4 kernel I Dyn4 output I Len4 pad)  //
{
    Choose(gabor, input, kernel, output, pad);
}
Declare(void, log, Dyn4 input I Dyn4 kernel I Dyn4 output I Len4 pad)  //
{
    Choose(log, input, kernel, output, pad);
}
Declare(void, zca_fit, Dyn2 input I Dyn1 mean I Dyn2 trans I Scalar epsilon I bool transform)  //
{
    Choose(zca_fit, input, mean, trans, epsilon, transform);
}
Declare(void, zca_trans, Dyn2 input I Dyn1 mean I Dyn2 trans)  //
{
    Choose(zca_trans, input, mean, trans);
}
Declare(void, zca_split, Dyn2 input I Dyn3 output)  //
{
    Choose(zca_split, input, output);
}
Declare(void, conv, Dyn4 input I Dyn4 kernel I Dyn4 output I Len2 stride I Len4 pad)  //
{
    Choose(conv, input, kernel, output, stride, pad);
}
Declare(void, fc, Dyn3 input I Dyn2 kernel I Dyn3 output)  //
{
    Choose(fc, input, kernel, output);
}
Declare(void, signfc, Dyn3 input I Dyn2 kernel I Dyn3 output)  //
{
    Choose(signfc, input, kernel, output);
}
Declare(void, pad, Dyn3 input I Dyn3 output I Len4 pad I Scalar value)  //
{
    Choose(pad, input, output, pad, value);
}
Declare(void, threshold, Dyn1 input I Scalar threshold I Scalar value)  //
{
    Choose(threshold, input, threshold, value);
}
Declare(void, quantize, Dyn1 input I Scalar lower I Scalar middle I Scalar upper)  //
{
    Choose(quantize, input, lower, middle, upper);
}
Declare(void, rank_code, Dyn2 input I Dyn3 output I bool sort)  //
{
    Choose(rank_code, input, output, sort);
}
Declare(void, rank_infinite, Dyn3 input I Scalar value)  //
{
    Choose(rank_infinite, input, value);
}
Declare(void, rank_fire, Dyn1 input I Dyn1 output I Scalar threshold)  //
{
    Choose(rank_fire, input, output, threshold);
}
Declare(void, rank_gather, Dyn3 input I Dyn2 output I Scalar threshold)
{
    Choose(rank_gather, input, output, threshold);
}
Declare(void, rank_scatter, Dyn2 input I Dyn3 output)  //
{
    Choose(rank_scatter, input, output);
}
Declare(void, rank_pool, Dyn4 input I Dyn4 output I Len2 kernel I Len2 stride I Len4 pad)  //
{
    Choose(rank_pool, input, output, kernel, stride, pad);
}
Declare(void, rank_inhibit, Dyn4 input I Scalar threshold)  //
{
    Choose(rank_inhibit, input, threshold);
}
Declare(Winners, rank_fcwta, Dyn3 input I Size radius I Size count I Scalar threshold)  //
{
    Choose(rank_fcwta, input, radius, count, threshold);
}
Declare(Winners, rank_convwta, Dyn5 input I Len2 radius I Size count I Scalar threshold)  //
{
    Choose(rank_convwta, input, radius, count, threshold);
}
Declare(void, rank_convstdp,
        Dyn5 input I Dyn4 kernel I Dyn5 output I const std::vector<STDPConfig> &config I const Winners &winners I Len2
            stride I Len4 pad)
{
    Choose(rank_convstdp, input, kernel, output, config, winners, stride, pad);
}
Declare(void, rank_fcstdp,
        Dyn3 input I Dyn2 kernel I Dyn3 output I const std::vector<STDPConfig> &config I const Winners &winners)
{
    Choose(rank_fcstdp, input, kernel, output, config, winners);
}
Declare(void, rate_code, Dyn2 input I Dyn3 output I bool sort)  //
{
    Choose(rate_code, input, output, sort);
}
Declare(void, rate_fire, Dyn3 input I Dyn3 output I Scalar threshold)  //
{
    Choose(rate_fire, input, output, threshold);
}
Declare(void, rate_gather, Dyn3 input I Dyn2 output I Scalar threshold)
{
    Choose(rate_gather, input, output, threshold);
}
Declare(void, rate_pool, Dyn5 input I Dyn4 rates I Dyn5 output I Len2 kernel I Len2 stride I Len4 pad)
{
    Choose(rate_pool, input, rates, output, kernel, stride, pad);
}
Declare(void, backward, Dyn2 input I Dyn2 output I Dyn1 target I Size time I Scalar gamma)
{
    Choose(backward, input, output, target, time, gamma);
}
Declare(void, labelize, Dyn3 input I Dyn1 output I Scalar threshold)  //
{
    Choose(labelize, input, output, threshold);
}
Declare(void, fcbackward, Dyn2 kernel I Dyn2 input I Dyn2 output I Dyn2 grad I Dyn2 next I BPConfig &config)
{
    Choose(fcbackward, kernel, input, output, grad, next, config);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Sparse5 sparse_alloc(Len5 len);

void sparse_dealloc(Sparse5 sparse);

Size sparse_elemsize(Sparse5 sparse);

Size sparse_memsize(Sparse5 sparse);

void sparse_copy(Sparse5 input, Sparse5 output);

void sparse_convert(Dyn5 input, Sparse5 output, Scalar threshold);

void sparse_convert(Sparse5 output, Dyn5 input);

void sparse_conv(Sparse5 input, Dyn4 kernel, Scalar threshold, Sparse5 output, Len2 stride, Len4 pad);

void sparse_pool(Sparse5 input, Sparse5 output, Len2 kernel, Len2 stride, Len4 pad);

void sparse_pad(Sparse5 input, Sparse5 output, Len4 pad);

void sparse_gather(Sparse5 input, Dyn4 output);

void sparse_code(Dyn4 input, Sparse5 output, bool sort);

void sparse_inhibit(Sparse5 input, Sparse5 output);

Winners sparse_convwta(Sparse5 input, Len2 radius, Size count);

void sparse_stdp(Sparse5 input, Dyn4 kernel, const std::vector<STDPConfig> &config, const Winners &winners, Len4 pad);
}  // namespace Core
}  // namespace Spyker

#undef I
#undef Declare
#undef Choose
