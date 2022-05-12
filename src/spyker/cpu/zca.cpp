#include "base.h"
//

#ifdef SPYKER_USE_BLAS
#include <blasw/blasw.h>
#endif

namespace Spyker
{
namespace Core
{
namespace CPU
{
#ifdef SPYKER_USE_BLAS
template <typename T>
void subtract(ARG1(T, input), ARG1(T, values), ARG1(T, output))
{
    VEC1(T, input) VEC1(T, values) VEC1(T, output);
    for (Size i = 0; i < input.x; ++i) output(i) = input(i) - values(i);
}

template <typename T>
void multiply(ARG1(T, input), ARG1(T, values), ARG1(T, output))
{
    VEC1(T, input) VEC1(T, values) VEC1(T, output);
    for (Size i = 0; i < input.x; ++i) output(i) = input(i) * values(i);
}

template <typename T>
void inverse(Vec1<T> values, T epsilon)
{
    for (Size i = 0; i < values.x; ++i)  //
        values(i) = T(1) / std::sqrt(std::max(T(0), values(i)) + epsilon);
}

template <typename T, typename F>
void zca_fit(Vec2<T> input, Vec1<T> mean, Vec2<T> trans, T epsilon, bool inplace, F lapack)
{
    auto middle = init<T>(input.y, input.x);
    auto sigma = init<T>(input.x, input.x);
    auto values = init<T>(input.x);
    auto vectors = init<T>(input.x, input.x);

    fill(Vec1<T>(middle.data, input.y), T(1));
    Blasw::dot(Blasw::rmat(input.data, input.y, input.x).trans(),  //
               Blasw::vec(middle.data, input.y),                   //
               Blasw::vec(mean.data, mean.x), 1 / T(input.y), 0);

#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)  //
        subtract(ARG(input(i)), ARG(mean), ARG(middle(i)));

    Blasw::update(Blasw::rmat(middle.data, middle.y, middle.x).trans(),  //
                  Blasw::rusym(sigma.data, sigma.x), 1 / T(input.y - 1), 0);

    Blasw::eigen(Blasw::rusym(sigma.data, sigma.x),  //
                 Blasw::vec(values.data, values.x), true, true);

    inverse(values, epsilon);

#pragma omp parallel for
    for (Size i = 0; i < sigma.y; ++i)  //
        multiply(ARG(sigma(i)), ARG(values), ARG(vectors(i)));

    Blasw::dot(Blasw::rmat(vectors.data, vectors.y, vectors.x),    //
               Blasw::rmat(sigma.data, sigma.y, sigma.x).trans(),  //
               Blasw::rmat(trans.data, trans.y, trans.x), 1, 0);

    if (inplace)
        Blasw::dot(Blasw::rmat(middle.data, middle.y, middle.x),  //
                   Blasw::rusym(trans.data, trans.x),             //
                   Blasw::rmat(input.data, input.y, input.x), 1, 0);

    deinit(middle, sigma, values, vectors);
}

template <typename T>
void zca_trans(Vec2<T> input, Vec1<T> mean, Vec2<T> trans)
{
    auto middle = init<T>(input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)  //
        subtract(ARG(input(i)), ARG(mean), ARG(middle(i)));

    Blasw::dot(Blasw::rmat(middle.data, middle.y, middle.x),  //
               Blasw::rusym(trans.data, trans.x),             //
               Blasw::rmat(input.data, input.y, input.x), 1, 0);

    deinit(middle);
}

#else

#define LAPACKE_sgesdd 0
#define LAPACKE_dgesdd 0

template <typename T, typename F>
void zca_fit(Vec2<T> input, Vec1<T> mean, Vec2<T> trans, T epsilon, bool inplace, F lapack)
{
    SpykerCompare(false, ==, true, "CPU::ZCA", "BLAS and LAPACK are not enabled in this build.");
}

template <typename T>
void zca_trans(Vec2<T> input, Vec1<T> mean, Vec2<T> trans)
{
    SpykerCompare(false, ==, true, "CPU::ZCA", "BLAS and LAPACK are not enabled in this build.");
}
#endif

template <typename T>
void zca_split(Size size, PTR(T, input), PTR(T, output1), PTR(T, output2))
{
    for (Size j = 0; j < size; ++j)
    {
        output1[j] = std::max(T(input[j]), T(0));
        output2[j] = std::max(T(-input[j]), T(0));
    }
}

template <typename T>
void zca_split(Vec2<T> input, Vec3<T> output)
{
#pragma omp parallel for
    for (Size i = 0; i < output.z; ++i) zca_split(input.x, input(i, 0), output(i, 0, 0), output(i, 1, 0));
}
}  // namespace CPU

void cpu_zca_fit(Dyn2 input, Dyn1 mean, Dyn2 trans, Scalar epsilon, bool transform)
{
    if (input.type == Type::F32)
        CPU::zca_fit<F32>(input, mean, trans, epsilon, transform, LAPACKE_sgesdd);
    else if (input.type == Type::F64)
        CPU::zca_fit<F64>(input, mean, trans, epsilon, transform, LAPACKE_dgesdd);
    else
        SpykerAssert(false, "CPU::ZCA", "Given type is not supported in this operation.");
}
void cpu_zca_trans(Dyn2 input, Dyn1 mean, Dyn2 trans)
{
    if (input.type == Type::F32)
        CPU::zca_trans<F32>(input, mean, trans);
    else if (input.type == Type::F64)
        CPU::zca_trans<F64>(input, mean, trans);
    else
        SpykerAssert(false, "CPU::ZCA", "Given type is not supported in this operation.");
}
void cpu_zca_split(Dyn2 input, Dyn3 output) { IfType(T, input.type, CPU::zca_split<T>(input, output)); }
}  // namespace Core
}  // namespace Spyker
