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

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <spyker/impl.h>

#include <memory>

#ifdef SPYKER_USE_CUDNN
#include <cudnn.h>
#endif

#undef IfReal

#define IfReal(name, type, expr) \
    if (type == Type::F16)       \
    {                            \
        using name = C16;        \
        expr;                    \
    }                            \
    else if (type == Type::F32)  \
    {                            \
        using name = F32;        \
        expr;                    \
    }                            \
    else if (type == Type::F64)  \
    {                            \
        using name = F64;        \
        expr;                    \
    }                            \
    else                         \
        SpykerAssert(false, "Core::Type", "Unknown type \"" << type << "\" given.")

#undef IfType

#define IfType(name, type, expr) \
    if (type == Type::I8)        \
    {                            \
        using name = I8;         \
        expr;                    \
    }                            \
    else if (type == Type::I16)  \
    {                            \
        using name = I16;        \
        expr;                    \
    }                            \
    else if (type == Type::I32)  \
    {                            \
        using name = I32;        \
        expr;                    \
    }                            \
    else if (type == Type::I64)  \
    {                            \
        using name = I64;        \
        expr;                    \
    }                            \
    else if (type == Type::U8)   \
    {                            \
        using name = U8;         \
        expr;                    \
    }                            \
    else if (type == Type::U16)  \
    {                            \
        using name = U16;        \
        expr;                    \
    }                            \
    else if (type == Type::U32)  \
    {                            \
        using name = U32;        \
        expr;                    \
    }                            \
    else if (type == Type::U64)  \
    {                            \
        using name = U64;        \
        expr;                    \
    }                            \
    else if (type == Type::F16)  \
    {                            \
        using name = C16;        \
        expr;                    \
    }                            \
    else if (type == Type::F32)  \
    {                            \
        using name = F32;        \
        expr;                    \
    }                            \
    else if (type == Type::F64)  \
    {                            \
        using name = F64;        \
        expr;                    \
    }                            \
    else                         \
        SpykerAssert(false, "Core::Type", "Unknown type \"" << type << "\" given.")

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define PTR(type, name) type *__restrict__ name

#define CudaCheck(expr)                                                                 \
    {                                                                                   \
        auto error = (expr);                                                            \
        SpykerCompare(error, ==, cudaSuccess, "Core::CUDA", cudaGetErrorString(error)); \
    }

#define CudnnCheck(expr)                                                                           \
    {                                                                                              \
        auto error = (expr);                                                                       \
        SpykerCompare(error, ==, CUDNN_STATUS_SUCCESS, "Core::cuDNN", cudnnGetErrorString(error)); \
    }

#define CublasCheck(expr)                                                                             \
    {                                                                                                 \
        auto error = (expr);                                                                          \
        SpykerCompare(error, ==, CUBLAS_STATUS_SUCCESS, "Core::cuBLAS", cublasGetErrorString(error)); \
    }

#define CurandCheck(expr)                                                                             \
    {                                                                                                 \
        auto error = (expr);                                                                          \
        SpykerCompare(error, ==, CURAND_STATUS_SUCCESS, "Core::cuRAND", curandGetErrorString(error)); \
    }

#define Thread1 Cize(256)

#define Index1 (blockIdx.x * Thread1 + threadIdx.x)

#define Config1(z, y, x) config1(z, y, x), Thread1

#define Thread2 Cize(16)

#define Index2Y (blockIdx.y * Thread2 + threadIdx.y)

#define Index2X (blockIdx.x * Thread2 + threadIdx.x)

#define Config2(z, y, x) config2(z, y, x), dim3(Thread2, Thread2, 1)

#define Thread1D Cize(256)

#define Block1D(type) Cize(Thread1D * 16 / sizeof(type))

#define Index1D(type) (blockIdx.x * Block1D(type) + threadIdx.x)

#define Config1D(type, z, y, x) config1d<type>(z, y, x), Thread1D

#define LimitsLine(type, minim, maxim)                                       \
    template <>                                                              \
    struct Limits<type>                                                      \
    {                                                                        \
        inline __device__ __host__ static type min() { return type(minim); } \
        inline __device__ __host__ static type max() { return type(maxim); } \
    };

namespace Spyker
{
struct C16
{
    __half data;

    inline __device__ C16() {}
    inline __device__ C16(__half value) : data(value) {}
    inline __host__ C16(Scalar value) : data(F32(value)) {}

    template <typename T>
    inline __device__ C16(T value)
    {
        data = F32(value);
    }
    template <typename T>
    inline __device__ C16 &operator=(T value)
    {
        data = F32(value);
        return *this;
    }
    template <typename T>
    inline __device__ operator T() const
    {
        return F32(data);
    }

    inline __device__ bool operator<(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data < value.data;
#else
        return F32(data) < F32(value.data);
#endif
    }
    inline __device__ bool operator<=(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data <= value.data;
#else
        return F32(data) <= F32(value.data);
#endif
    }
    inline __device__ bool operator>(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data > value.data;
#else
        return F32(data) > F32(value.data);
#endif
    }
    inline __device__ bool operator>=(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data >= value.data;
#else
        return F32(data) >= F32(value.data);
#endif
    }

    inline __device__ C16 operator+(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data + value.data;
#else
        return F32(data) + F32(value.data);
#endif
    }
    inline __device__ C16 operator-(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data - value.data;
#else
        return F32(data) - F32(value.data);
#endif
    }
    inline __device__ C16 operator*(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data * value.data;
#else
        return F32(data) * F32(value.data);
#endif
    }
    inline __device__ C16 operator/(C16 value) const
    {
#if __CUDA_ARCH__ >= 530
        return data / value.data;
#else
        return F32(data) / F32(value.data);
#endif
    }

    inline __device__ C16 &operator+=(C16 value)
    {
#if __CUDA_ARCH__ >= 530
        data += value.data;
#else
        data = F32(data) + F32(value.data);
#endif
        return *this;
    }
    inline __device__ C16 &operator-=(C16 value)
    {
#if __CUDA_ARCH__ >= 530
        data -= value.data;
#else
        data = F32(data) - F32(value.data);
#endif
        return *this;
    }
    inline __device__ C16 &operator*=(C16 value)
    {
#if __CUDA_ARCH__ >= 530
        data *= value.data;
#else
        data = F32(data) * F32(value.data);
#endif
        return *this;
    }
    inline __device__ C16 &operator/=(C16 value)
    {
#if __CUDA_ARCH__ >= 530
        data /= value.data;
#else
        data = F32(data) / F32(value.data);
#endif
        return *this;
    }

    inline __device__ C16 operator-() const
    {
#if __CUDA_ARCH__ >= 530
        return -data;
#else
        return -F32(data);
#endif
        return *this;
    }
};

inline __device__ C16 cmax(C16 first, C16 second)
{
#if __CUDA_ARCH__ >= 800
    return __hmax(first.data, second.data);
#else
    return max(F32(first), F32(second));
#endif
}
inline __device__ C16 cmin(C16 first, C16 second)
{
#if __CUDA_ARCH__ >= 800
    return __hmin(first.data, second.data);
#else
    return min(F32(first), F32(second));
#endif
}

CreateLimits(LimitsLine);

template <>
struct Limits<C16>
{
    inline __device__ static C16 min() { return C16(-6.550400e+004); }
    inline __device__ static C16 max() { return C16(6.550400e+004); }
};

template <>
struct ToFloat<C16>
{
    using Type = C16;
};

inline Dyn1 todyn(Vec1<C16> input)  //
{
    return Dyn1(input.data, Type::F16, input.x);
}
inline Dyn2 todyn(Vec2<C16> input)  //
{
    return Dyn2(input.data, Type::F16, input.y, input.x);
}
inline Dyn3 todyn(Vec3<C16> input)  //
{
    return Dyn3(input.data, Type::F16, input.z, input.y, input.x);
}
inline Dyn4 todyn(Vec4<C16> input)  //
{
    return Dyn4(input.data, Type::F16, input.t, input.z, input.y, input.x);
}
inline Dyn5 todyn(Vec5<C16> input)  //
{
    return Dyn5(input.data, Type::F16, input.u, input.t, input.z, input.y, input.x);
}

namespace Core
{
template <typename T>
Type TypeName()
{
    if (std::is_same<T, I8>()) return Type::I8;
    if (std::is_same<T, I16>()) return Type::I16;
    if (std::is_same<T, I32>()) return Type::I32;
    if (std::is_same<T, I64>()) return Type::I64;
    if (std::is_same<T, U8>()) return Type::U8;
    if (std::is_same<T, U16>()) return Type::U16;
    if (std::is_same<T, U32>()) return Type::U32;
    if (std::is_same<T, U64>()) return Type::U64;
    if (std::is_same<T, C16>()) return Type::F16;
    if (std::is_same<T, F32>()) return Type::F32;
    if (std::is_same<T, F64>()) return Type::F64;
    SpykerAssert(false, "Interface::Type", "Given type is not recognized.");
}

namespace CUDA
{
using Cize = int;

inline dim3 config1(Size z, Size y, Size x)
{
    const Cize block = Thread1;
    return dim3((x + block - 1) / block, y, z);
}

inline dim3 config2(Size z, Size y, Size x)
{
    const Cize block = Thread2;
    return dim3((x + block - 1) / block, (y + block - 1) / block, z);
}

template <typename T>
dim3 config1d(Size z, Size y, Size x)
{
    const Cize block = Block1D(T);
    return dim3((x + block - 1) / block, y, z);
}

template <typename T>
Vec1<T> init(Size x)
{
    return Vec1<T>((T *)cuda_alloc(sizeof(T) * x), x);
}
template <typename T>
Vec2<T> init(Size y, Size x)
{
    return Vec2<T>((T *)cuda_alloc(sizeof(T) * y * x), y, x);
}
template <typename T>
Vec3<T> init(Size z, Size y, Size x)
{
    return Vec3<T>((T *)cuda_alloc(sizeof(T) * z * y * x), z, y, x);
}
template <typename T>
Vec4<T> init(Size t, Size z, Size y, Size x)
{
    return Vec4<T>((T *)cuda_alloc(sizeof(T) * t * z * y * x), t, z, y, x);
}
template <typename T>
Vec5<T> init(Size u, Size t, Size z, Size y, Size x)
{
    return Vec5<T>((T *)cuda_alloc(sizeof(T) * u * t * z * y * x), u, t, z, y, x);
}

template <typename V>
void deinit(V vec)
{
    cuda_dealloc(vec.data);
}
template <typename H, typename... T>
void deinit(H head, T &&...tail)
{
    cuda_dealloc(head.data);
    deinit(std::forward<T>(tail)...);
}
template <typename V>
void fill(V vec, typename V::Type value)
{
    cuda_fill(vec.size(), Dyn(vec.data, TypeName<typename V::Type>()), Scalar(value));
}
template <typename type>
type d2h(type *input)
{
    type output;
    cuda2cpu(sizeof(type), input, &output);
    return output;
}
template <typename type>
void h2d(type input, type *output)
{
    cpu2cuda(sizeof(type), &input, output);
}

template <typename I, typename O>
void copy(Vec1<I> input, Vec1<O> output)
{
    cuda_cast(input.size(), Dyn(input.data, TypeName<I>()), Dyn(output.data, TypeName<O>()));
}
template <typename I, typename O>
void copy(Vec2<I> input, Vec2<O> output)
{
    cuda_cast(input.size(), Dyn(input.data, TypeName<I>()), Dyn(output.data, TypeName<O>()));
}
template <typename I, typename O>
void copy(Vec3<I> input, Vec3<O> output)
{
    cuda_cast(input.size(), Dyn(input.data, TypeName<I>()), Dyn(output.data, TypeName<O>()));
}
template <typename I, typename O>
void copy(Vec4<I> input, Vec4<O> output)
{
    cuda_cast(input.size(), Dyn(input.data, TypeName<I>()), Dyn(output.data, TypeName<O>()));
}
template <typename I, typename O>
void copy(Vec5<I> input, Vec5<O> output)
{
    cuda_cast(input.size(), Dyn(input.data, TypeName<I>()), Dyn(output.data, TypeName<O>()));
}

void *_maxval(Dyn2 input, void *data);

void *_minval(Dyn2 input, void *data);

U32 *_maxidx(Dyn2 input, U32 *index, void *data);

template <typename T>
Size maxsize(Size size)
{
    Size output = 0;
    const Size block = Block1D(T);

    while (true)
    {
        size = (size + block - 1) / block;
        output += size;
        if (size == 1) return output;
    }
}
template <typename T>
Vec1<T> maxval(Vec2<T> input, T *data)
{
    data = (T *)_maxval(todyn(input), data);
    return Vec1<T>(data, input.y);
}
template <typename T>
Vec1<T> minval(Vec2<T> input, T *data)
{
    data = (T *)_minval(todyn(input), data);
    return Vec1<T>(data, input.y);
}
template <typename T>
Vec1<U32> maxidx(Vec2<T> input, U32 *index, T *data)
{
    index = _maxidx(todyn(input), index, data);
    return Vec1<U32>(index, input.y);
}

void sync();

const char *curandGetErrorString(curandStatus_t error);

const char *cublasGetErrorString(cublasStatus_t error);

struct cublas
{
    cublasHandle_t handle;
    inline cublas() { CublasCheck(cublasCreate(&handle)); }
    inline ~cublas() { cublasDestroy(handle); }
};

extern std::unique_ptr<cublas> cublas_static;

#ifdef SPYKER_USE_CUDNN
struct cudnn
{
    cudnnHandle_t handle;
    inline cudnn() { CudnnCheck(cudnnCreate(&handle)); }
    inline ~cudnn() { cudnnDestroy(handle); }
};

extern std::unique_ptr<cudnn> cudnn_static;
#endif
}  // namespace CUDA
}  // namespace Core
}  // namespace Spyker
