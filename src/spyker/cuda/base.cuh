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
    if (type == Type::F32)       \
    {                            \
        using name = F32;        \
        expr;                    \
    }                            \
    else if (type == Type::F64)  \
    {                            \
        using name = F64;        \
        expr;                    \
    }                            \
    SpykerAssert(false, "Core::Type", "Unknown type \"" << type << "\" given.")

#undef IfType

#define IfType IfNotHalf

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

#define LimitsLine(type, minim, maxim)                                \
    template <>                                                       \
    struct Limits<type>                                               \
    {                                                                 \
        __device__ __host__ static type min() { return type(minim); } \
        __device__ __host__ static type max() { return type(maxim); } \
    };

namespace Spyker
{
using C16 = __half;

CreateLimits(LimitsLine);

template <>
struct ToFloat<C16>
{
    using Type = C16;
};

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

template <typename T1, typename T2>
inline __device__ void cast(T1 input, T2 &output)
{
    output = input;
}
template <typename T1>
inline __device__ void cast(T1 input, C16 &output)
{
    output = F32(input);
}
template <typename T2>
inline __device__ void cast(C16 input, T2 &output)
{
    output = F32(input);
}
inline __device__ void cast(C16 input, C16 &output) { output = input; }

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
