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

#define HALF_ENABLE_CPP11_CSTDINT 1
#include <half.hpp>
#include <numeric>

namespace std
{
using half_float::abs;
using half_float::acos;
using half_float::acosh;
using half_float::asin;
using half_float::asinh;
using half_float::atan;
using half_float::atan2;
using half_float::atanh;
using half_float::cbrt;
using half_float::ceil;
using half_float::copysign;
using half_float::cos;
using half_float::cosh;
using half_float::erf;
using half_float::erfc;
using half_float::exp;
using half_float::exp2;
using half_float::expm1;
using half_float::fabs;
using half_float::fdim;
using half_float::floor;
using half_float::fma;
using half_float::fmax;
using half_float::fmin;
using half_float::fmod;
using half_float::fpclassify;
using half_float::frexp;
using half_float::hypot;
using half_float::ilogb;
using half_float::isfinite;
using half_float::isgreater;
using half_float::isgreaterequal;
using half_float::isinf;
using half_float::isless;
using half_float::islessequal;
using half_float::islessgreater;
using half_float::isnan;
using half_float::isnormal;
using half_float::isunordered;
using half_float::ldexp;
using half_float::lgamma;
using half_float::llrint;
using half_float::llround;
using half_float::log;
using half_float::log10;
using half_float::log1p;
using half_float::log2;
using half_float::logb;
using half_float::lrint;
using half_float::lround;
using half_float::modf;
using half_float::nanh;
using half_float::nearbyint;
using half_float::nextafter;
using half_float::nexttoward;
using half_float::pow;
using half_float::remainder;
using half_float::remquo;
using half_float::rint;
using half_float::round;
using half_float::rsqrt;
using half_float::scalbln;
using half_float::scalbn;
using half_float::signbit;
using half_float::sin;
using half_float::sinh;
using half_float::sqrt;
using half_float::tan;
using half_float::tanh;
using half_float::tgamma;
using half_float::trunc;

template <typename T>
T pow2(T data)
{
    return data * data;
}
}  // namespace std
