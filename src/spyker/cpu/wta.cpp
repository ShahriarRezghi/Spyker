#include "base.h"
//

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename T, typename E = void>
struct Sum
{
};
template <typename T>
struct Sum<T, typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, F16>::value>::type>
{
    using Type = T;

    static void rank_fcwta(ARG2(T, input), ARG1(T, temp), T threshold)
    {
        VEC2(T, input) VEC1(T, temp);

        T max = maxval(input) * T(2);
        fill(temp, Limits<T>::min());
        for (Size i = input.y - 1; i >= 0; --i)
        {
            T *id = input(i, 0);
            T top = T(max * (input.y - i - 1));
            for (Size j = 0; j < input.x; ++j)
                if (id[j] > threshold) temp(j) = id[j] + top;
        }
    }
};
template <typename T>
struct Sum<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
    using Type = U16;

    template <typename S>
    static void rank_fcwta(ARG2(T, input), ARG1(S, temp), T)
    {
        VEC2(T, input) VEC1(S, temp);

        fill(temp, S(0));
        for (Size i = input.y - 1; i >= 0; --i)
        {
            S time = input.y - i;
            T *id = input(i, 0);
            for (Size j = 0; j < input.x; ++j)
                if (id[j]) temp(j) = time;
        }
    }
};

template <typename T>
bool rank_fcwta(Vec1<T> temp, Size radius, Winner &winner)
{
    winner = {0, 0, 0, 0};
    T min = Limits<T>::min();
    Size idx = maxidx(temp);
    if (std::abs(temp(idx) - min) < Epsilon) return false;

    winner.x = idx;
    Size s = std::max(Size(0), winner.x - radius);
    Size e = std::min(temp.x, winner.x + radius + 1);
    fill(Vec1<T>(&temp(s), e - s), min);
    return true;
}

template <typename T>
Winners rank_fcwta(Vec3<T> input, Size radius, Size count, T threshold)
{
    using S = typename Sum<T>::Type;

    Winners winners(input.z);
    auto temp = init<S>(input.z, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)
    {
        winners[i].reserve(count);
        Sum<T>::rank_fcwta(ARG(input(i)), ARG(temp(i)), threshold);

        for (Size j = 0; j < count; ++j)
        {
            Winner winner;
            if (!rank_fcwta(temp(i), radius, winner)) break;
            winners[i].push_back(winner);
        }
    }

    deinit(temp);
    return winners;
}

template <typename T>
bool rank_convwta(Vec3<T> temp, Len2 radius, Winner &winner)
{
    winner = {0, 0, 0, 0};
    T min = Limits<T>::min();
    Size idx = maxidx(temp);
    if (std::abs(temp.data[idx] - min) < Epsilon) return false;

    Size len = temp.y * temp.x;
    winner.z = idx / len;
    winner.y = (idx % len) / temp.x;
    winner.x = (idx % len) % temp.x;
    Size sj = std::max(Size(0), winner.y - radius.y);
    Size ej = std::min(temp.y, winner.y + radius.y + 1);
    Size sk = std::max(Size(0), winner.x - radius.x);
    Size ek = std::min(temp.x, winner.x + radius.x + 1);

    for (Size i = 0; i < temp.z; ++i)
        for (Size j = sj; j < ej; ++j)  //
            fill(Vec1<T>(temp(i, j, sk), ek - sk), min);

    fill(temp(winner.z), min);
    return true;
}

template <typename T>
Winners rank_convwta(Vec5<T> input, Len2 radius, Size count, T threshold)
{
    using S = typename Sum<T>::Type;

    BatchSize(input.u);
    Winners winners(input.u);
    auto temp = init<S>(batch_size, input.z, input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
    {
        Size batch_index = BatchIndex(i);

        winners[i].reserve(count);
        Len2 len(input.t, input.z * input.y * input.x);
        Sum<T>::rank_fcwta(ARG(Vec2<T>(input(i).data, len)),  //
                           ARG(Vec1<S>(temp(batch_index).data, len.x)), threshold);

        for (Size j = 0; j < count; ++j)
        {
            Winner winner;
            if (!rank_convwta(temp(batch_index), radius, winner)) break;
            winners[i].push_back(winner);
        }
    }

    deinit(temp);
    return winners;
}
}  // namespace CPU

Winners cpu_rank_fcwta(Dyn3 input, Size radius, Size count, Scalar threshold)
{
    IfType(T, input.type, return CPU::rank_fcwta<T>(input, radius, count, threshold));
}
Winners cpu_rank_convwta(Dyn5 input, Len2 radius, Size count, Scalar threshold)
{
    IfType(T, input.type, return CPU::rank_convwta<T>(input, radius, count, threshold));
}
}  // namespace Core
}  // namespace Spyker
