// BSD 3-Clause License
//
// Copyright (c) 2022, University of Tehran (Shahriar Rezghi <shahriar25.ss@gmail.com>)
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

#include <omp.h>
#include <spyker/impl.h>

#ifdef SPYKER_USE_BLAS
#include <blasw/blasw.h>
#endif

#ifdef _WIN32
#define PTR(type, name) type *__restrict name
#else
#define PTR(type, name) type *__restrict__ name
#endif

#define ARG(name) name.data, name.len()

#define ARG1(type, name) PTR(type, _##name##p), Size _##name##l
#define ARG2(type, name) PTR(type, _##name##p), Len2 _##name##l
#define ARG3(type, name) PTR(type, _##name##p), Len3 _##name##l
#define ARG4(type, name) PTR(type, _##name##p), Len4 _##name##l
#define ARG5(type, name) PTR(type, _##name##p), Len5 _##name##l

#define VEC1(type, name) Vec1<type> name(_##name##p, _##name##l);
#define VEC2(type, name) Vec2<type> name(_##name##p, _##name##l);
#define VEC3(type, name) Vec3<type> name(_##name##p, _##name##l);
#define VEC4(type, name) Vec4<type> name(_##name##p, _##name##l);
#define VEC5(type, name) Vec5<type> name(_##name##p, _##name##l);

#define LimitsLine(type, minim, maxim)            \
    template <>                                   \
    struct Limits<type>                           \
    {                                             \
        static type min() { return type(minim); } \
        static type max() { return type(maxim); } \
    };

#define BatchSize(size)                                 \
    bool thread_filled = size >= omp_get_max_threads(); \
    Size batch_size = std::min<Size>(omp_get_max_threads(), size);

#define BatchIndex(index) (Select<Size>(index, omp_get_thread_num())[thread_filled])

namespace Spyker
{
namespace Core
{
CreateLimits(LimitsLine);

template <typename T>
struct Select
{
    T D[2];
    Select(T A, T B) : D{A, B} {}
    T operator[](bool I) { return D[I]; }
};

namespace CPU
{
template <typename T>
Vec1<T> init(Size x)
{
    return Vec1<T>((T *)cpu_alloc(sizeof(T) * x), x);
}
template <typename T>
Vec2<T> init(Size y, Size x)
{
    return Vec2<T>((T *)cpu_alloc(sizeof(T) * y * x), y, x);
}
template <typename T>
Vec3<T> init(Size z, Size y, Size x)
{
    return Vec3<T>((T *)cpu_alloc(sizeof(T) * z * y * x), z, y, x);
}
template <typename T>
Vec4<T> init(Size t, Size z, Size y, Size x)
{
    return Vec4<T>((T *)cpu_alloc(sizeof(T) * t * z * y * x), t, z, y, x);
}
template <typename T>
Vec5<T> init(Size u, Size t, Size z, Size y, Size x)
{
    return Vec5<T>((T *)cpu_alloc(sizeof(T) * u * t * z * y * x), u, t, z, y, x);
}

template <typename V>
void deinit(V vec)
{
    cpu_dealloc(vec.data);
}
template <typename H, typename... T>
void deinit(H head, T &&...tail)
{
    cpu_dealloc(head.data);
    deinit(std::forward<T>(tail)...);
}
template <typename V1, typename V2>
void copy(V1 input, V2 output)
{
    std::copy(input.begin(), input.end(), output.begin());
}
template <typename V>
void fill(V vec, typename V::Type value)
{
    std::fill(vec.begin(), vec.end(), value);
}
template <typename V>
Size maxidx(V vec)
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}
template <typename V>
Size minidx(V vec)
{
    return std::distance(vec.begin(), std::min_element(vec.begin(), vec.end()));
}
template <typename T>
T maxval(Size size, PTR(T, input))
{
    T max = input[0];
    for (Size i = 0; i < size; ++i) max = std::max(max, input[i]);
    return max;
}
template <typename V>
typename V::Type maxval(V vec)
{
    return maxval(vec.size(), vec.data);
}
}  // namespace CPU
}  // namespace Core
}  // namespace Spyker

#ifdef SPYKER_USE_DNNL
#include <dnnl.hpp>

namespace Spyker
{
namespace Core
{
namespace CPU
{
struct onednn
{
    dnnl::engine engine;
    dnnl::stream stream;

    onednn()
    {
        engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
        stream = dnnl::stream(engine);
    }
};

extern std::unique_ptr<onednn> onednn_static;
}  // namespace CPU
}  // namespace Core
}  // namespace Spyker
#endif
