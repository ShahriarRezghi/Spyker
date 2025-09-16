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

#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename T>
void step1(Vec3<T> input_, Vec3<F32> output_)
{
    float kernel_[] = {
        0.0030, 0.0133, 0.0219, 0.0133, 0.0030,  //
        0.0133, 0.0596, 0.0983, 0.0596, 0.0133,  //
        0.0219, 0.0983, 0.1621, 0.0983, 0.0219,  //
        0.0133, 0.0596, 0.0983, 0.0596, 0.0133,  //
        0.0030, 0.0133, 0.0219, 0.0133, 0.0030,  //
    };

    auto kernel = init<F32>(1, 1, 5, 5);
    cpu2cuda(kernel.size() * sizeof(F32), kernel_, kernel.data);
    Vec4<T> input = {input_.data, input_.z, 1, input_.y, input_.x};
    Vec4<F32> output = {output_.data, output_.z, 1, output_.y, output_.x};
    cuda_conv(todyn(input), todyn(kernel), todyn(output), {1, 1}, {2, 2, 2, 2});
    deinit(kernel);
}

template <typename T>
void step2(Vec3<T> input_, Vec4<F32> output)
{
    float kernel_[] = {
        -1, -1, 0,  1,  1,  //
        -2, -2, 0,  2,  2,  //
        -3, -6, 0,  6,  3,  //
        -2, -2, 0,  2,  2,  //
        -1, -1, 0,  1,  1,  //

        1,  2,  3,  2,  1,   //
        1,  2,  6,  2,  1,   //
        0,  0,  0,  0,  0,   //
        -1, -2, -6, -2, -1,  //
        -1, -2, -3, -2, -1,  //

    };

    auto kernel = init<F32>(2, 1, 5, 5);
    cpu2cuda(kernel.size() * sizeof(F32), kernel_, kernel.data);
    Vec4<F32> input = {input_.data, input_.z, 1, input_.y, input_.x};
    cuda_conv(todyn(input), todyn(kernel), todyn(output), {1, 1}, {2, 2, 2, 2});
    deinit(kernel);
}

__global__ void cuda_step3(Cize size, PTR(F32, input), PTR(F32, ouput))
{
    F32 *X = input + blockIdx.y * 2 * size, *Y = X + size;
    F32 *value = ouput + blockIdx.y * 2 * size, *angle = value + size;
    Cize i = Index1D(F32);
    if (size <= i) return;

    value[i] = sqrt(X[i] * X[i] + Y[i] * Y[i]);
    angle[i] = atan2(Y[i], X[i]);
}

void step3(Vec4<F32> input, Vec4<F32> output)
{
    Cize size = input.y * input.x;
    cuda_step3<<<Config1D(F32, 1, input.t, size)>>>(size, input.data, output.data);
}

__device__ F32 step4(Cize size, PTR(F32, input), F32 angle, Cize i, Cize j)
{
    angle *= 180 / PI;
    if (angle < 0) angle += 180;
    F32 v = input[i * size + j], v1, v2;

    if (angle < 22.5 || 157.5 <= angle)
    {
        v1 = input[i * size + (j + 1)];
        v2 = input[i * size + (j - 1)];
    }
    else if (angle < 67.5)
    {
        v1 = input[(i + 1) * size + (j - 1)];
        v2 = input[(i - 1) * size + (j + 1)];
    }
    else if (angle < 112.5)
    {
        v1 = input[(i + 1) * size + j];
        v2 = input[(i - 1) * size + j];
    }
    else
    {
        v1 = input[(i - 1) * size + (j - 1)];
        v2 = input[(i + 1) * size + (j + 1)];
    }
    return v >= v1 && v >= v2 ? v : 0;
}

__global__ void step4(Cize Y, Cize X, PTR(F32, id), PTR(F32, od))
{
    od += blockIdx.z * Y * X;
    F32 *vd = id + blockIdx.z * 2 * Y * X, *ad = vd + Y * X;
    Cize i = Index2Y, j = Index2X;
    if (Y <= i || X <= j) return;

    if (i == 0 || j == 0 || i == Y - 1 || j == X - 1)
        od[i * X + j] = 0;
    else
        od[i * X + j] = step4(X, vd, ad[i * X + j], i, j);
}

void step4(Vec4<F32> input, Vec3<F32> output)
{
    step4<<<Config2(output.z, output.y, output.x)>>>(input.y, input.x, input.data, output.data);
}

#define WEAK 1
#define STRONG 2

__device__ U8 step5(U8 data, U8 low, U8 high)
{
    return high <= data ? STRONG : (low <= data ? WEAK : 0);  //
}

__global__ void step5(Cize Y, Cize X, PTR(F32, input), PTR(U8, output), PTR(F32, maxim), U8 low, U8 high)
{
    input += blockIdx.z * Y * X, output += blockIdx.z * Y * X;
    Cize i = Index2Y, j = Index2X;
    if (Y <= i || X <= j) return;

    F32 div = 255 / (maxim[blockIdx.z] - Epsilon);
    U8 value = input[i * X + j] * div, stat = step5(value, low, high);

    if (stat == WEAK && i != 0 && j != 0 && i != Y - 1 && j != X - 1)
        if (U8(input[(i - 1) * X + (j - 1)] * div) >= high ||  //
            U8(input[(i - 1) * X + (j + 0)] * div) >= high ||  //
            U8(input[(i - 1) * X + (j + 1)] * div) >= high ||  //
            U8(input[(i + 0) * X + (j - 1)] * div) >= high ||  //
            U8(input[(i + 0) * X + (j + 1)] * div) >= high ||  //
            U8(input[(i + 1) * X + (j - 1)] * div) >= high ||  //
            U8(input[(i + 1) * X + (j + 0)] * div) >= high ||  //
            U8(input[(i + 1) * X + (j + 1)] * div) >= high)
            value = 255;

    output[i * X + j] = (stat == STRONG ? 255 : value);
}

void step5(Vec3<F32> input, Vec3<U8> output, Vec1<F32> max, U8 low, U8 high)
{
    step5<<<Config2(input.z, input.y, input.x)>>>(input.y, input.x, input.data, output.data, max.data, low, high);
}

template <typename T>
void canny(Vec3<T> input, Vec3<U8> output, U8 low, U8 high)
{
    auto temp2 = init<F32>(input.z, 2, input.y, input.x);
    auto temp3 = init<F32>(input.z, 2, input.y, input.x);
    Vec3<F32> temp1 = {temp3.data, input.z, input.y, input.x};
    Vec3<F32> temp4 = {temp2.data, input.z, input.y, input.x};

    step1(input, temp1);
    step2(temp1, temp2);
    step3(temp2, temp3);
    step4(temp3, temp4);
    auto max = maxval(Vec2<F32>{temp4.data, temp4.z, temp4.y * temp4.x}, temp3.data);
    step5(temp4, output, max, low, high);

    deinit(temp2, temp3);
}
}  // namespace CUDA

void cuda_canny(Dyn3 input, Dyn3 output, Scalar low, Scalar high)
{
    IfType(T, input.type, CUDA::canny<T>(input, output, low, high));
}
}  // namespace Core
}  // namespace Spyker
