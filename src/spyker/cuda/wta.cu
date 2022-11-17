#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename T, typename S>
__global__ void select(Cize Y, Cize X, PTR(T, input), PTR(S, temp))
{
    input += blockIdx.y * Y * X, temp += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    S value = 0;
    for (Cize i = Y - 1; i >= 0; --i)  //
        if (input[i * X + j]) value = Y - i;
    temp[j] = value;
}

template <typename T>
__global__ void select(Cize Y, Cize X, PTR(T, id), PTR(T, od), PTR(T, min), PTR(T, max), T thresh)
{
    id += blockIdx.y * Y * X, od += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    T value = Limits<T>::min();
    T minim = min[blockIdx.y], diff = (max[blockIdx.y] - minim) * T(2);
    for (Cize i = Y - 1; i >= 0; --i)
    {
        Cize idx = i * X + j;
        T level = T(Y - i - 1);
        if (id[idx] > thresh) value = level + (id[idx] - minim) / diff;
    }
    od[j] = value;
}

template <typename T, typename E = void>
struct Sum
{
};
template <typename T>
struct Sum<T, typename std::enable_if<std::is_floating_point<T>::value ||  //
                                      std::is_same<T, F16>::value || std::is_same<T, C16>::value>::type>
{
    using Type = T;

    static void select(Vec3<T> input, Vec2<T> temp, T threshold)
    {
        auto value = init<T>(2, input.z, maxsize<T>(input(0).size()));
        auto min = minval(Vec2<T>(input.data, input.z, input(0).size()), value(0).data);
        auto max = maxval(Vec2<T>(input.data, input.z, input(0).size()), value(1).data);
        CUDA::select<<<Config1(1, input.z, input.x)>>>(  //
            input.y, input.x, input.data, temp.data, min.data, max.data, threshold);
        deinit(value);
    }
};
template <typename T>
struct Sum<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
    using Type = U16;

    static void select(Vec3<T> input, Vec2<Type> temp, T)
    {
        CUDA::select<<<Config1(1, input.z, input.x)>>>(input.y, input.x, input.data, temp.data);
    }
};

struct Index
{
    I32 si, sj, ej, sk, ek;
};

template <typename T>
__global__ void select(Cize isize, Cize osize, PTR(T, output), PTR(Index, index))
{
    output += blockIdx.y * osize + index[blockIdx.y].si;
    Cize idx = Index1D(T), end = min(isize, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = Limits<T>::min();
}

template <typename T>
__global__ void select(Cize Z, Cize Y, Cize X, PTR(T, output), PTR(Index, idx))
{
    Cize z = Index1;
    output += (blockIdx.y * Z + z) * Y * X;
    if (Z <= z) return;

    Index index = idx[blockIdx.y];
    for (Cize i = index.sj; i < index.ej; ++i)
        for (Cize j = index.sk; j < index.ek; ++j)  //
            output[i * X + j] = Limits<T>::min();
}

bool select(Len3 temp, Index &start, U32 index, Len2 radius, Winner &winner)
{
    if (index == U32(-1)) return false;
    Size size = temp.y * temp.x;
    winner.z = index / size;
    winner.y = (index % size) / temp.x;
    winner.x = (index % size) % temp.x;

    start.si = winner.z * size;
    start.sj = std::max<I32>(0, winner.y - radius.y);
    start.ej = std::min(temp.y, winner.y + radius.y + 1);
    start.sk = std::max<I32>(0, winner.x - radius.x);
    start.ek = std::min(temp.x, winner.x + radius.x + 1);
    return true;
}

template <typename T>
void select(Vec4<T> temp, Vec1<Index> start, Len2 radius)
{
    select<<<Config1D(T, 1, temp.t, temp.y * temp.x)>>>(temp(0).size(), temp.y * temp.x, temp.data, start.data);
    select<<<Config1(1, temp.t, temp.z)>>>(temp.z, temp.y, temp.x, temp.data, start.data);
}

template <typename T>
Winners select(Vec5<T> input, Len2 radius, Size count, T threshold)
{
    using S = typename Sum<T>::Type;

    auto temp = init<S>(input.u, input.z, input.y, input.x);
    auto maxi = init<U32>(temp.t, maxsize<S>(temp(0).size()));
    auto maxv = init<S>(temp.t, maxsize<S>(temp(0).size()));
    auto start = init<Index>(input.u);

    Winners winners(input.u);
    std::vector<U32> idx_(input.u);
    std::vector<Index> start_(start.x);
    Sum<T>::select(Vec3<T>{input.data, input.u, input.t, input.z * input.y * input.x},
                   Vec2<S>{temp.data, temp.t, temp.z * temp.y * temp.x}, threshold);

    for (Size _ = 0; _ < count; ++_)
    {
        auto idx = maxidx(Vec2<S>(temp.data, temp.t, temp(0).size()), maxi.data, maxv.data);
        cuda2cpu(idx.size() * sizeof(U32), idx.data, idx_.data());
        for (Size i = 0; i < input.u; ++i)
        {
            Winner winner = {0, 0, 0, 0};
            if (select(temp(i).len(), start_[i], idx_[i], radius, winner)) winners[i].push_back(winner);
        }
        cpu2cuda(start.size() * sizeof(Index), start_.data(), start.data);
        select(temp, start, radius);
    }

    deinit(temp, maxi, maxv, start);
    return winners;
}
}  // namespace CUDA

Winners cuda_rank_fcwta(Dyn3 input, Size radius, Size count, Scalar threshold)
{
    Dyn5 temp(input.data, input.type, input.z, input.y, 1, 1, input.x);
    return cuda_rank_convwta(temp, {0, radius}, count, threshold);
}
Winners cuda_rank_convwta(Dyn5 input, Len2 radius, Size count, Scalar threshold)
{
    IfType(T, input.type, return CUDA::select<T>(input, radius, count, threshold));
}
}  // namespace Core
}  // namespace Spyker
