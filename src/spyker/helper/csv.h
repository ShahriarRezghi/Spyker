// BSD 3-Clause License
//
// Copyright (c) 2022, Shahriar Rezghi <shahriar25.ss@gmail.com>,
//                     Mohammad-Reza A. Dehaqani <dehaqani@ut.ac.ir>,
//                     University of Tehran
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

#include <fstream>
#include <string>
#include <vector>

namespace Spyker
{
namespace Helper
{
/// CSV file reader
class SpykerExport CSV
{
    std::ifstream stream;
    std::string line, delim;

    bool _readline(std::vector<std::string> &row);

public:
    /// @param path path of the CSV file to be read.
    /// @param delim delimiter of the CSV file that separates values.
    CSV(const std::string &path, const std::string &delim = ", ");

    /// Read next line from the CSV file.
    ///
    /// If the line is empty, it is skipped until a non-empty line is read.
    ///
    /// @param[out] row vector of strings that will be filled with values from the line.
    /// @return success or failure to read. False value indicates end of the file or read error.
    bool readline(std::vector<std::string> &row);

    /// Skip the specified number of lines.
    ///
    /// @param lines number of lines to be skipped.
    Size skip(Size lines);
};
}  // namespace Helper
}  // namespace Spyker
