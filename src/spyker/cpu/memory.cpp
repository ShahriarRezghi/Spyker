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

#include "base.h"
//
#include <cstdlib>
#include <cstring>

namespace Spyker
{
namespace Core
{
void* cpu_alloc(Size size)
{
#ifdef _WIN32
    return malloc(size);
#else
    return aligned_alloc(256, size);
#endif
}

void cpu_dealloc(void* data) { std::free(data); }

void cpu_copy(Size size, void* input, void* output) { std::memcpy(output, input, size); }

template <typename T1, typename T2>
void cpu_cast(Size size, T1* input, T2* output)
{
    std::copy(input, input + size, output);
}

void cpu_cast(Size size, Dyn input, Dyn output)
{
    IfType(T1, input.type, IfType(T2, output.type, cpu_cast(size, input.to<T1>(), output.to<T2>())));
}

template <typename T>
void cpu_fill(Size size, T* data, T value)
{
    std::fill(data, data + size, value);
}

void cpu_fill(Size size, Dyn data, Scalar value) { IfType(T, data.type, cpu_fill<T>(size, data.to<T>(), value)); }

namespace CPU
{
#ifdef SPYKER_USE_DNNL
std::unique_ptr<onednn> onednn_static;
#endif
}  // namespace CPU
}  // namespace Core
}  // namespace Spyker
