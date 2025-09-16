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

#include "mnist.h"

#include <cstring>
#include <fstream>

namespace Spyker
{
namespace Helper
{
int32_t readInt(std::ifstream &file)
{
    int32_t data;
    auto ptr = (uint8_t *)&data;
    file.read((std::ifstream::char_type *)ptr, sizeof(int32_t));
    data = (ptr[3] << 0) | (ptr[2] << 8) | (ptr[1] << 16) | (ptr[0] << 24);
    return data;
}

Tensor mnistLabel(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open()) return Tensor();
    if (readInt(file) != 2049) return Tensor();
    Size size = readInt(file);
    Tensor data(Type::U8, {size});
    file.read((std::ifstream::char_type *)data.data(), data.numel());
    return file.fail() ? Tensor() : data;
}

Tensor mnistData(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open()) return Tensor();
    if (readInt(file) != 2051) return Tensor();
    Size size = readInt(file), rows = readInt(file), cols = readInt(file);
    Tensor data(Type::U8, {size, 1, rows, cols});
    file.read((std::ifstream::char_type *)data.data(), data.numel());
    return file.fail() ? Tensor() : data;
}
}  // namespace Helper
}  // namespace Spyker
