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

#include "csv.h"

namespace Spyker
{
namespace Helper
{
CSV::CSV(const std::string &path, const std::string &delim) : stream(path), delim(delim) {}

bool CSV::_readline(std::vector<std::string> &row)
{
    row.clear();
    if (!std::getline(stream, line)) return false;
    if (line.empty()) return true;

    size_t prev = 0, index = 0;
    while (index != std::string::npos)
    {
        auto end = (index = line.find(delim, prev));
        if (index != std::string::npos) end -= prev;
        row.push_back(line.substr(prev, end));
        prev = index + 1;
    }
    return true;
}

bool CSV::readline(std::vector<std::string> &row)
{
    while (true)
    {
        if (!_readline(row)) return false;
        if (!line.empty()) return true;
    }
}

Size CSV::skip(Size lines)
{
    Size skiped = 0;
    for (; skiped < lines; ++skiped)
        if (!std::getline(stream, line)) break;
    return skiped;
}
}  // namespace Helper
}  // namespace Spyker
