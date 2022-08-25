#include "base.h"
//

namespace Spyker
{
namespace Core
{
void normalize(PTR(F64, input), Size size)
{
    F64 sum = 0;
    for (Size i = 0; i < size; ++i) sum += input[i];
    for (Size i = 0; i < size; ++i) input[i] /= sum;
}

void normal_kernel(Vec1<F64> input, F64 mean, F64 std)
{
    std::normal_distribution<F64> dist(mean, std);
    for (Size i = 0; i < input.x; ++i) input(i) = dist(Generator);
}

void gaussian_kernel(Vec2<F64> kernel, F64 std)
{
    Size size = kernel.x / 2;
    for (Size i = -size; i <= size; ++i)
        for (Size j = -size; j <= size; ++j)
        {
            F64 dist = std::sqrt(i * i + j * j);
            F64 value = std::exp(-.5 * std::pow(dist / std, 2));
            *kernel(i + size, j + size) = value;
        }
    normalize(kernel.data, kernel.size());
}

void gabor_kernel(Vec2<F64> kernel, F64 sigma, F64 theta, F64 gamma, F64 lambda, F64 psi)
{
    Size size = kernel.x / 2;
    for (Size i = -size; i <= size; ++i)
        for (Size j = -size; j <= size; ++j)
        {
            F64 x = i * std::cos(theta) + j * std::sin(theta);
            F64 y = -i * std::sin(theta) + j * std::cos(theta);
            F64 A = std::exp(-(x * x + gamma * gamma * y * y) / (2 * sigma * sigma));
            F64 B = std::cos(2 * PI * x / lambda + psi);
            *kernel(i + size, j + size) = A * B;
        }
    normalize(kernel.data, kernel.size());
}

void log_kernel(Vec2<F64> kernel, F64 std)
{
    Size size = kernel.x / 2;
    F64 div = -PI * std::pow(std, 4);
    for (Size i = -size; i <= size; ++i)
        for (Size j = -size; j <= size; ++j)
        {
            F64 A = (i * i + j * j) / (-2 * std * std);
            F64 value = (1 + A) * std::exp(A) / div;
            *kernel(i + size, j + size) = value;
        }
}

void chw2hwc(Len3 chw, PTR(U8, input), PTR(U8, output))
{
    Size c = chw.z, h = chw.y, w = chw.x;

#pragma omp parallel for collapse(2)
    for (Size i = 0; i < h; ++i)
        for (Size j = 0; j < w; ++j)
        {
            U8* id = input + i * w + j;
            U8* od = output + (i * w + j) * c;
            for (Size k = 0; k < c; ++k) od[k] = id[k * h * w];
        }
}

void hwc2chw(Len3 chw, U8* input, U8* output)
{
    Size c = chw.z, h = chw.y, w = chw.x;

#pragma omp parallel for collapse(2)
    for (Size i = 0; i < h; ++i)
        for (Size j = 0; j < w; ++j)
        {
            U8* id = input + (i * w + j) * c;
            U8* od = output + i * w + j;
            for (Size k = 0; k < c; ++k) od[k * h * w] = id[k];
        }
}

Vec2<U16> poisson_create(Size time, Size bins)
{
    auto ztp = [](U16 k, F32 a) {
        F32 p = 1 / (std::exp(a) - 1);
        for (Size i = 1; i <= k; ++i) p *= a / i;
        return p;
    };
    Vec2<U16> data = CPU::init<U16>(time, bins);

#pragma omp parallel for
    for (Size i = 0; i < time; ++i)
    {
        F32 scale = i < 1 ? 5 : 4;
        U16 lambda = i + 1, range = std::ceil(scale * std::sqrt(lambda));
        U16 start = std::max(1, lambda - range), end = lambda + range;

        Size index = 0;
        U16* dd = data(i, 0);
        for (U16 j = start; j <= end; ++j)
        {
            Size count = std::nearbyint(ztp(j, lambda) * bins);
            if (index + count > bins) count = bins - index;
            CPU::fill(Vec1<U16>(dd + index, count), j);
            index += count;
        }
        CPU::fill(Vec1<U16>(dd + index, bins - index), lambda);
        std::random_shuffle(dd, dd + bins);
    }
    return data;
}
}  // namespace Core
}  // namespace Spyker
