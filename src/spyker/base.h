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

#include <spyker/utils.h>

#include <cfloat>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>

#define Comma ,

#define PI 3.14159265358979323846

namespace Spyker
{
template <typename T>
void print(T &&value)
{
    std::cout << value << " " << std::endl;
}
template <typename Head, typename... Tail>
void print(Head &&head, Tail &&...tail)
{
    std::cout << head << " ";
    print(std::forward<Tail>(tail)...);
}
inline void print() { std::cout << std::endl; }

inline void format(std::string text) { print(text); }

template <typename Head, typename... Tail>
void format(std::string text, Head &&head, Tail &&...tail)
{
    std::smatch match;
    if (std::regex_search(text, match, std::regex("\\{[0-9]?\\}")))
    {
        std::ostringstream os;
        auto sub = match.str();
        if (std::regex_match(sub, std::regex("\\{[0-9]\\}")))  //
            os << std::fixed << std::setprecision(sub[1] - '0');
        os << head;
        text.replace(match.position(), match.length(), os.str());
    }
    format(text, std::forward<Tail>(tail)...);
}

struct Timer
{
    std::chrono::high_resolution_clock::time_point time;

    inline Timer() { reset(); }

    inline float elapsed()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - time);
        time = now;
        return float(elapsed.count());
    }

    inline void reset() { time = std::chrono::high_resolution_clock::now(); }
};

struct Len2
{
    Size y, x;

    inline Len2() {}

    inline Len2(Size y, Size x) : y(y), x(x) {}

    inline Len2(const Shape &shape)
    {
        SpykerCompare(shape.size(), ==, 2, "Core::Length", "Input shape must be two dimensional.");
        y = shape[0], x = shape[1];
    }
};

struct Len3
{
    Size z, y, x;

    inline Len3() {}

    inline Len3(Size z, Size y, Size x) : z(z), y(y), x(x) {}

    inline Len3(const Shape &shape)
    {
        SpykerCompare(shape.size(), ==, 3, "Core::Length", "Input shape must be three dimensional.");
        z = shape[0], y = shape[1], x = shape[2];
    }
};

struct Len4
{
    Size t, z, y, x;

    inline Len4() {}

    inline Len4(Size t, Size z, Size y, Size x) : t(t), z(z), y(y), x(x) {}

    inline Len4(const Shape &shape)
    {
        SpykerCompare(shape.size(), ==, 4, "Core::Length", "Input shape must be four dimensional.");
        t = shape[0], z = shape[1], y = shape[2], x = shape[3];
    }
};

struct Len5
{
    Size u, t, z, y, x;

    inline Len5() {}

    inline Len5(Size u, Size t, Size z, Size y, Size x) : u(u), t(t), z(z), y(y), x(x) {}

    inline Len5(const Shape &shape)
    {
        SpykerCompare(shape.size(), ==, 5, "Core::Length", "Input shape must be five dimensional.");
        u = shape[0], t = shape[1], z = shape[2], y = shape[3], x = shape[4];
    }
};

template <typename T>
struct Vec1
{
    using Type = T;

    T *data = nullptr;
    Size x = -1;

    Vec1() {}

    Vec1(T *data, Size x) : data(data), x(x) {}

    T &operator()(Size i) { return data[i]; }

    Size len() { return x; }

    Size size() { return x; }

    T *begin() { return data; }

    T *end() { return data + x; }
};

template <typename T>
struct Vec2
{
    using Type = T;

    T *data = nullptr;
    Size y = -1, x = -1;

    Vec2() {}

    Vec2(T *data, Len2 len) : data(data), y(len.y), x(len.x) {}

    Vec2(T *data, Size y, Size x) : data(data), y(y), x(x) {}

    Vec1<T> operator()(Size i) { return Vec1<T>(data + i * x, x); }

    T *operator()(Size i, Size j) { return data + i * x + j; }

    Len2 len() { return Len2(y, x); }

    Size size() { return y * x; }

    T *begin() { return data; }

    T *end() { return data + y * x; }

    Vec1<T> flat() { return Vec1<T>(data, y * x); }
};

template <typename T>
struct Vec3
{
    using Type = T;

    T *data = nullptr;
    Size z = -1, y = -1, x = -1;

    Vec3() {}

    Vec3(T *data, Len3 len) : data(data), z(len.z), y(len.y), x(len.x) {}

    Vec3(T *data, Size z, Size y, Size x) : data(data), z(z), y(y), x(x) {}

    Vec2<T> operator()(Size i) { return Vec2<T>(data + i * y * x, y, x); }

    Vec1<T> operator()(Size i, Size j) { return Vec1<T>(data + (i * y + j) * x, x); }

    T *operator()(Size i, Size j, Size k) { return data + (i * y + j) * x + k; }

    Len3 len() { return Len3(z, y, x); }

    Size size() { return z * y * x; }

    T *begin() { return data; }

    T *end() { return data + z * y * x; }

    Vec1<T> flat() { return Vec1<T>(data, z * y * x); }
};

template <typename T>
struct Vec4
{
    using Type = T;

    T *data = nullptr;
    Size t = -1, z = -1, y = -1, x = -1;

    Vec4() {}

    Vec4(T *data, Len4 len) : data(data), t(len.t), z(len.z), y(len.y), x(len.x) {}

    Vec4(T *data, Size t, Size z, Size y, Size x) : data(data), t(t), z(z), y(y), x(x) {}

    Vec3<T> operator()(Size i) { return Vec3<T>(data + i * z * y * x, z, y, x); }

    Vec2<T> operator()(Size i, Size j) { return Vec2<T>(data + (i * z + j) * y * x, y, x); }

    Vec1<T> operator()(Size i, Size j, Size k) { return Vec1<T>(data + ((i * z + j) * y + k) * x, x); }

    T *operator()(Size i, Size j, Size k, Size l) { return data + ((i * z + j) * y + k) * x + l; }

    Len4 len() { return Len4(t, z, y, x); }

    Size size() { return t * z * y * x; }

    T *begin() { return data; }

    T *end() { return data + t * z * y * x; }

    Vec1<T> flat() { return Vec1<T>(data, t * z * y * x); }
};

template <typename T>
struct Vec5
{
    using Type = T;

    T *data = nullptr;
    Size u = -1, t = -1, z = -1, y = -1, x = -1;

    Vec5() {}

    Vec5(T *data, Len5 len) : data(data), u(len.u), t(len.t), z(len.z), y(len.y), x(len.x) {}

    Vec5(T *data, Size u, Size t, Size z, Size y, Size x) : data(data), u(u), t(t), z(z), y(y), x(x) {}

    Vec4<T> operator()(Size i) { return Vec4<T>(data + i * t * z * y * x, t, z, y, x); }

    Vec3<T> operator()(Size i, Size j) { return Vec3<T>(data + (i * t + j) * z * y * x, z, y, x); }

    Vec2<T> operator()(Size i, Size j, Size k) { return Vec2<T>(data + ((i * t + j) * z + k) * y * x, y, x); }

    Vec1<T> operator()(Size i, Size j, Size k, Size l)
    {
        return Vec2<T>(data + (((i * t + j) * z + k) * y + l) * x, x);
    }

    T *operator()(Size i, Size j, Size k, Size l, Size m) { return data + (((i * t + j) * z + k) * y + l) * x + m; }

    Len5 len() { return Len5(u, t, z, y, x); }

    Size size() { return u * t * z * y * x; }

    T *begin() { return data; }

    T *end() { return data + u * t * z * y * x; }

    Vec1<T> flat() { return Vec1<T>(data, u * t * z * y * x); }
};

inline std::ostream &operator<<(std::ostream &os, Timer &timer)  //
{
    return os << "Elapsed: " << timer.elapsed();
}

template <typename T>
std::ostream &operator<<(std::ostream &os, Vec1<T> vec)
{
    os << "[";
    for (Size i = 0; i < vec.x; ++i)  //
        os << F64(vec(i)) << (i == vec.x - 1 ? "" : ",");
    return os << "]";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, Vec2<T> vec)
{
    os << "[";
    for (Size i = 0; i < vec.y; ++i)
    {
        for (Size j = 0; j < vec.x; ++j)  //
            os << F64(*vec(i, j)) << (j == vec.x - 1 ? "" : ",");
        os << std::endl;
    }
    return os << "]";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, Vec3<T> vec)
{
    for (Size i = 0; i < vec.z; ++i)
    {
        for (Size j = 0; j < vec.y; ++j)
        {
            for (Size k = 0; k < vec.x; ++k)  //
                os << F64(*vec(i, j, k)) << (k == vec.x - 1 ? "" : ",");
            os << std::endl;
        }
        os << std::endl;
    }
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i)  //
        os << F64(vec[i]) << (i == vec.size() - 1 ? "" : ",");
    return os << "]";
}

struct Dyn
{
    void *data;
    Type type;

    inline Dyn(void *data, Type type) : data(data), type(type) {}

    template <typename T>
    T *to()
    {
        return (T *)data;
    }
};

struct Dyn1
{
    void *data;
    Type type;
    Size x;

    inline Dyn1(void *data, Type type, Size x) : data(data), type(type), x(x) {}

    inline Dyn1(const Tensor &tensor)
    {
        auto shape = tensor.shape();
        SpykerCompare(tensor.dims(), ==, 1, "Core::Dynamic", "Input tensor must be one dimensional.");
        data = tensor.data(), type = tensor.type(), x = shape[0];
    }

    template <typename T>
    operator Vec1<T>() const
    {
        return Vec1<T>((T *)data, x);
    }
};

struct Dyn2
{
    void *data;
    Type type;
    Size y, x;

    inline Dyn2(void *data, Type type, Size y, Size x) : data(data), type(type), y(y), x(x) {}

    inline Dyn2(const Tensor &tensor)
    {
        auto shape = tensor.shape();
        SpykerCompare(tensor.dims(), ==, 2, "Core::Dynamic", "Input tensor must be two dimensional.");
        data = tensor.data(), type = tensor.type(), y = shape[0], x = shape[1];
    }

    template <typename T>
    operator Vec2<T>() const
    {
        return Vec2<T>((T *)data, y, x);
    }
};

struct Dyn3
{
    void *data;
    Type type;
    Size z, y, x;

    inline Dyn3(void *data, Type type, Size z, Size y, Size x) : data(data), type(type), z(z), y(y), x(x) {}

    inline Dyn3(const Tensor &tensor)
    {
        auto shape = tensor.shape();
        SpykerCompare(tensor.dims(), ==, 3, "Core::Dynamic", "Input tensor must be three dimensional.");
        data = tensor.data(), type = tensor.type(), z = shape[0], y = shape[1], x = shape[2];
    }

    template <typename T>
    operator Vec3<T>() const
    {
        return Vec3<T>((T *)data, z, y, x);
    }
};

struct Dyn4
{
    void *data;
    Type type;
    Size t, z, y, x;

    inline Dyn4(void *data, Type type, Size t, Size z, Size y, Size x) : data(data), type(type), t(t), z(z), y(y), x(x)
    {
    }

    inline Dyn4(const Tensor &tensor)
    {
        auto shape = tensor.shape();
        SpykerCompare(tensor.dims(), ==, 4, "Core::Dynamic", "Input tensor must be four dimensional.");
        data = tensor.data(), type = tensor.type(), t = shape[0], z = shape[1], y = shape[2], x = shape[3];
    }

    template <typename T>
    operator Vec4<T>() const
    {
        return Vec4<T>((T *)data, t, z, y, x);
    }
};

struct Dyn5
{
    void *data;
    Type type;
    Size u, t, z, y, x;

    inline Dyn5(void *data, Type type, Size u, Size t, Size z, Size y, Size x)
        : data(data), type(type), u(u), t(t), z(z), y(y), x(x)
    {
    }

    inline Dyn5(const Tensor &tensor)
    {
        auto shape = tensor.shape();
        SpykerCompare(tensor.dims(), ==, 5, "Core::Dynamic", "Input tensor must be five dimensional.");
        data = tensor.data(), type = tensor.type();
        u = shape[0], t = shape[1], z = shape[2], y = shape[3], x = shape[4];
    }

    template <typename T>
    operator Vec5<T>() const
    {
        return Vec5<T>((T *)data, u, t, z, y, x);
    }
};

struct Spridx
{
    U8 z, y, x;
    inline Spridx(U8 z, U8 y, U8 x) : z(z), y(y), x(x) {}
};

struct Sparse3
{
    Spridx *data = nullptr;
    Size size = 0, max = 16;

    inline Spridx *begin() { return data; }
    inline Spridx *end() { return data + size; }
    inline Spridx &operator()(Size index) { return data[index]; }

    inline void resize(Size size)
    {
        max = size;
        data = (Spridx *)realloc(data, max * sizeof(Spridx));
    }
    inline void add(Spridx index)
    {
        if (max <= size) resize(max * 2);
        data[size] = index, ++size;
    }
};

struct Sparse5
{
    Sparse3 *data = nullptr;
    Size u = 0, t = 0, z = 0, y = 0, x = 0;

    inline Len5 len() { return {u, t, z, y, x}; }
    inline Size size() { return u * t * z * y * x; }
    inline Sparse3 &operator()(Size i, Size j) { return data[i * t + j]; }
};

inline bool operator==(Len2 A, Len2 B) { return A.y == B.y && A.x == B.x; }
inline bool operator==(Len3 A, Len3 B) { return A.z == B.z && A.y == B.y && A.x == B.x; }
inline bool operator==(Len4 A, Len4 B) { return A.t == B.t && A.z == B.z && A.y == B.y && A.x == B.x; }
inline bool operator==(Len5 A, Len5 B) { return A.u == B.u && A.t == B.t && A.z == B.z && A.y == B.y && A.x == B.x; }

template <typename T>
Dyn1 todyn(Vec1<T> input)
{
    return Dyn1(input.data, TypeName<T>(), input.x);
}
template <typename T>
Dyn2 todyn(Vec2<T> input)
{
    return Dyn2(input.data, TypeName<T>(), input.y, input.x);
}
template <typename T>
Dyn3 todyn(Vec3<T> input)
{
    return Dyn3(input.data, TypeName<T>(), input.z, input.y, input.x);
}
template <typename T>
Dyn4 todyn(Vec4<T> input)
{
    return Dyn4(input.data, TypeName<T>(), input.t, input.z, input.y, input.x);
}
template <typename T>
Dyn5 todyn(Vec5<T> input)
{
    return Dyn5(input.data, TypeName<T>(), input.u, input.t, input.z, input.y, input.x);
}

template <typename T>
struct ToFloat
{
    using Type = F32;
};
template <>
struct ToFloat<F16>
{
    using Type = F16;
};
template <>
struct ToFloat<F64>
{
    using Type = F64;
};

extern std::mt19937 Generator;

const static F16 HLF_MIN = std::numeric_limits<F16>::min();
const static F16 HLF_MAX = std::numeric_limits<F16>::max();
}  // namespace Spyker

#define IfReal(name, type, expr) \
    if (type == Type::F16)       \
    {                            \
        using name = F16;        \
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

#define IfInt(name, type, expr) \
    if (type == Type::I8)       \
    {                           \
        using name = I8;        \
        expr;                   \
    }                           \
    else if (type == Type::I16) \
    {                           \
        using name = I16;       \
        expr;                   \
    }                           \
    else if (type == Type::I32) \
    {                           \
        using name = I32;       \
        expr;                   \
    }                           \
    else if (type == Type::I64) \
    {                           \
        using name = I64;       \
        expr;                   \
    }                           \
    else if (type == Type::U8)  \
    {                           \
        using name = U8;        \
        expr;                   \
    }                           \
    else if (type == Type::U16) \
    {                           \
        using name = U16;       \
        expr;                   \
    }                           \
    else if (type == Type::U32) \
    {                           \
        using name = U32;       \
        expr;                   \
    }                           \
    else if (type == Type::U64) \
    {                           \
        using name = U64;       \
        expr;                   \
    }                           \
    else                        \
        SpykerAssert(false, "Core::Type", "Unknown type \"" << type << "\" given.")

#define IfNotHalf(name, type, expr) \
    if (type == Type::I8)           \
    {                               \
        using name = I8;            \
        expr;                       \
    }                               \
    else if (type == Type::I16)     \
    {                               \
        using name = I16;           \
        expr;                       \
    }                               \
    else if (type == Type::I32)     \
    {                               \
        using name = I32;           \
        expr;                       \
    }                               \
    else if (type == Type::I64)     \
    {                               \
        using name = I64;           \
        expr;                       \
    }                               \
    else if (type == Type::U8)      \
    {                               \
        using name = U8;            \
        expr;                       \
    }                               \
    else if (type == Type::U16)     \
    {                               \
        using name = U16;           \
        expr;                       \
    }                               \
    else if (type == Type::U32)     \
    {                               \
        using name = U32;           \
        expr;                       \
    }                               \
    else if (type == Type::U64)     \
    {                               \
        using name = U64;           \
        expr;                       \
    }                               \
    else if (type == Type::F32)     \
    {                               \
        using name = F32;           \
        expr;                       \
    }                               \
    else if (type == Type::F64)     \
    {                               \
        using name = F64;           \
        expr;                       \
    }                               \
    else                            \
        SpykerAssert(false, "Core::Type", "Unknown type \"" << type << "\" given.")

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
        using name = F16;        \
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

#define CreateLimits(Macro)                                        \
    template <typename T>                                          \
    struct Limits                                                  \
    {                                                              \
    };                                                             \
    Macro(I8, -128LL, 127LL);                                      \
    Macro(I16, -32768LL, 32767LL);                                 \
    Macro(I32, -2147483648LL, 2147483647LL);                       \
    Macro(I64, -9223372036854775807LL - 1, 9223372036854775807LL); \
    Macro(U8, 0ULL, 255ULL);                                       \
    Macro(U16, 0ULL, 65535ULL);                                    \
    Macro(U32, 0ULL, 4294967295ULL);                               \
    Macro(U64, 0ULL, 18446744073709551615ULL);                     \
    Macro(F16, HLF_MIN, HLF_MAX);                                  \
    Macro(F32, -FLT_MAX, FLT_MAX);                                 \
    Macro(F64, -DBL_MAX, DBL_MAX);

#define Epsilon 1e-5f
