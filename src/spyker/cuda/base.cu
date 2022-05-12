#include <cuda_fp16.h>

#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
void sync()
{
    CudaCheck(cudaGetLastError());
    CudaCheck(cudaDeviceSynchronize());
}

const char *curandGetErrorString(curandStatus_t error)
{
    if (error == CURAND_STATUS_SUCCESS) return "CURAND_STATUS_SUCCESS";
    if (error == CURAND_STATUS_VERSION_MISMATCH) return "CURAND_STATUS_VERSION_MISMATCH";
    if (error == CURAND_STATUS_NOT_INITIALIZED) return "CURAND_STATUS_NOT_INITIALIZED";
    if (error == CURAND_STATUS_ALLOCATION_FAILED) return "CURAND_STATUS_ALLOCATION_FAILED";
    if (error == CURAND_STATUS_TYPE_ERROR) return "CURAND_STATUS_TYPE_ERROR";
    if (error == CURAND_STATUS_OUT_OF_RANGE) return "CURAND_STATUS_OUT_OF_RANGE";
    if (error == CURAND_STATUS_LENGTH_NOT_MULTIPLE) return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    if (error == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED) return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    if (error == CURAND_STATUS_LAUNCH_FAILURE) return "CURAND_STATUS_LAUNCH_FAILURE";
    if (error == CURAND_STATUS_PREEXISTING_FAILURE) return "CURAND_STATUS_PREEXISTING_FAILURE";
    if (error == CURAND_STATUS_INITIALIZATION_FAILED) return "CURAND_STATUS_INITIALIZATION_FAILED";
    if (error == CURAND_STATUS_ARCH_MISMATCH) return "CURAND_STATUS_ARCH_MISMATCH";
    if (error == CURAND_STATUS_INTERNAL_ERROR) return "CURAND_STATUS_INTERNAL_ERROR";
    return "Unknown";
}

const char *cublasGetErrorString(cublasStatus_t error)
{
    if (error == CUBLAS_STATUS_SUCCESS) return "CUBLAS_STATUS_SUCCESS";
    if (error == CUBLAS_STATUS_NOT_INITIALIZED) return "CUBLAS_STATUS_NOT_INITIALIZED";
    if (error == CUBLAS_STATUS_ALLOC_FAILED) return "CUBLAS_STATUS_ALLOC_FAILED";
    if (error == CUBLAS_STATUS_INVALID_VALUE) return "CUBLAS_STATUS_INVALID_VALUE";
    if (error == CUBLAS_STATUS_ARCH_MISMATCH) return "CUBLAS_STATUS_ARCH_MISMATCH";
    if (error == CUBLAS_STATUS_MAPPING_ERROR) return "CUBLAS_STATUS_MAPPING_ERROR";
    if (error == CUBLAS_STATUS_EXECUTION_FAILED) return "CUBLAS_STATUS_EXECUTION_FAILED";
    if (error == CUBLAS_STATUS_INTERNAL_ERROR) return "CUBLAS_STATUS_INTERNAL_ERROR";
    if (error == CUBLAS_STATUS_NOT_SUPPORTED) return "CUBLAS_STATUS_NOT_SUPPORTED";
    if (error == CUBLAS_STATUS_LICENSE_ERROR) return "CUBLAS_STATUS_LICENSE_ERROR";
    return "Unknown";
}

std::unique_ptr<cublas> cublas_static;

#ifdef SPYKER_USE_CUDNN
std::unique_ptr<cudnn> cudnn_static;
#endif

template <typename T>
__device__ T maxv(T first, T second)
{
    return max(first, second);
}
__device__ C16 maxv(C16 first, C16 second) { return C16(max(F32(first), F32(second))); }

template <typename T>
__global__ void maxval(Cize isize, Cize osize, PTR(T, input), PTR(T, output))
{
    input += blockIdx.y * isize, output += blockIdx.y * osize;
    __shared__ T temp[Thread1D];
    temp[threadIdx.x] = Limits<T>::min();

    Cize idx = Index1D(T), end = min(isize, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) temp[threadIdx.x] = maxv(temp[threadIdx.x], input[i]);

    for (Cize i = Thread1D / 2; i > 0; i >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < i) temp[threadIdx.x] = maxv(temp[threadIdx.x], temp[threadIdx.x + i]);
    }
    if (threadIdx.x == 0) output[blockIdx.x] = temp[0];
}

template <typename T>
Vec1<T> maxval_(Vec2<T> input, T *data)
{
    Vec2<T> max = {data, input.y, 0};
    while (true)
    {
        max.x = (input.x + Block1D(T) - 1) / Block1D(T);
        maxval<<<Config1D(T, 1, input.y, input.x)>>>(input.x, max.x, input.data, max.data);
        if (max.x == 1) return {max.data, max.y};
        input = max, max.data += max.size();
    }
}

template <typename T>
__device__ T minv(T first, T second)
{
    return min(first, second);
}
__device__ C16 minv(C16 first, C16 second) { return C16(min(F32(first), F32(second))); }

template <typename T>
__global__ void minval(Cize isize, Cize osize, PTR(T, input), PTR(T, output))
{
    input += blockIdx.y * isize, output += blockIdx.y * osize;
    __shared__ T temp[Thread1D];
    temp[threadIdx.x] = Limits<T>::max();

    Cize idx = Index1D(T), end = min(isize, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) temp[threadIdx.x] = minv(temp[threadIdx.x], input[i]);

    for (Cize i = Thread1D / 2; i > 0; i >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < i) temp[threadIdx.x] = minv(temp[threadIdx.x], temp[threadIdx.x + i]);
    }
    if (threadIdx.x == 0) output[blockIdx.x] = temp[0];
}

template <typename T>
Vec1<T> minval_(Vec2<T> input, T *data)
{
    Vec2<T> max = {data, input.y, 0};
    while (true)
    {
        max.x = (input.x + Block1D(T) - 1) / Block1D(T);
        minval<<<Config1D(T, 1, input.y, input.x)>>>(input.x, max.x, input.data, max.data);
        if (max.x == 1) return {max.data, max.y};
        input = max, max.data += max.size();
    }
}

template <typename T>
__global__ void cuda_maxidx(Cize isize, Cize osize, PTR(U32, iindex), PTR(U32, oindex), PTR(T, ivalue), PTR(T, ovalue))
{
    if (iindex != nullptr) iindex += blockIdx.y * isize;
    oindex += blockIdx.y * osize;
    ivalue += blockIdx.y * isize;
    ovalue += blockIdx.y * osize;

    __shared__ U32 tindex[Thread1D];
    __shared__ T tvalue[Thread1D];

    tindex[threadIdx.x] = U32(-1), tvalue[threadIdx.x] = Limits<T>::min();
    Cize idx = Index1D(T), end = min(isize, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D)
        if (ivalue[i] > tvalue[threadIdx.x])
        {
            tvalue[threadIdx.x] = ivalue[i];
            tindex[threadIdx.x] = (iindex == nullptr ? i : iindex[i]);
        }

    for (Cize i = Thread1D / 2; i > 0; i >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < i)
            if (tvalue[threadIdx.x + i] > tvalue[threadIdx.x])
            {
                tvalue[threadIdx.x] = tvalue[threadIdx.x + i];
                tindex[threadIdx.x] = tindex[threadIdx.x + i];
            }
    }
    if (threadIdx.x == 0) oindex[blockIdx.x] = tindex[0], ovalue[blockIdx.x] = tvalue[0];
}

template <typename T>
Vec1<U32> maxidx_(Vec2<T> input_, U32 *index, T *data)
{
    T *ivalue = input_.data;
    Vec2<U32> max = {index, input_.y, 0};
    Vec2<U32> input = {nullptr, input_.y, input_.x};

    while (true)
    {
        max.x = (input.x + Block1D(T) - 1) / Block1D(T);
        cuda_maxidx<<<Config1D(T, 1, input.y, input.x)>>>(input.x, max.x, input.data, max.data, ivalue, data);
        if (max.x == 1) return {max.data, max.y};
        input = max, ivalue = data, max.data += max.size(), data += max.size();
    }
}

void *_maxval(Dyn2 input, void *data)  //
{
    IfType(T, input.type, return maxval_<T>(input, (T *)data).data);
}
void *_minval(Dyn2 input, void *data)  //
{
    IfType(T, input.type, return minval_<T>(input, (T *)data).data);
}
U32 *_maxidx(Dyn2 input, U32 *index, void *data)
{
    IfType(T, input.type, return maxidx_<T>(input, index, (T *)data).data);
}
}  // namespace CUDA
}  // namespace Core
}  // namespace Spyker
