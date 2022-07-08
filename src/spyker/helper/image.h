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

#include <string>

namespace Spyker
{
namespace Helper
{
/// Read Image from file.
///
/// @param path path of the file that will be read.
/// @param mode channel mode of reading. values can be 'Y', 'YA', 'RGB', 'RGBA' or
/// empty string for the original channels (default empty).
/// @param size width and height of the output image to be scaled. pass {-1, -1} for original size (default {-1, -1}).
/// @return image read as a tensor. failure to read returns empty tensor.
SpykerExport Tensor readImage(const std::string& path, std::string mode = "", Shape size = {-1, -1});

/// Read Image from file.
///
/// @param[out] output tensor that will be written to.
/// @param path path of the file that will be read.
/// @param mode channel mode of reading. values are 'Y', 'YA', 'RGB', 'RGBA' or
/// empty string for the original channels (default empty).
/// @param size width and height of the output image to be scaled. pass {-1, -1} for original size (default {-1, -1}).
/// @return success or failure to read.
SpykerExport bool readImage(Tensor output, const std::string& path, std::string mode = "", Shape size = {-1, -1});

/// Write image to file.
///
/// @param input input tensor that will be written to file.
/// @param path path of the file that will be written.
/// @param format format of the image file. values can be 'PNG', 'BMP', 'TGA', 'JPG'.
/// @return success or failure to write.
SpykerExport bool writeImage(Tensor input, const std::string& path, std::string format);
}  // namespace Helper
}  // namespace Spyker
