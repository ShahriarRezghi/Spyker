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

#include "shape.h"

namespace Spyker
{
Shape convShape(Shape input, Shape kernel, Shape stride, Shape pad)
{
    SpykerCompare(input.size(), ==, 5, "Shape::Conv", "Input dimensions must be 5.");
    SpykerCompare(kernel.size(), ==, 4, "Shape::Conv", "Kernel dimensions must be 4.");
    SpykerCompare(stride.size(), ==, 2, "Shape::Conv", "Stride dimensions must be 2.");
    SpykerCompare(pad.size(), ==, 4, "Shape::Conv", "Pad dimensions must be 4.");
    SpykerCompare(input[2], ==, kernel[1], "Shape::Conv", "Input and kernel channels are not compatible.");

    Shape output = {input[0], input[1], kernel[0], 0, 0};
    output[3] = (input[3] + pad[0] + pad[2] - kernel[2]) / stride[0] + 1;
    output[4] = (input[4] + pad[1] + pad[3] - kernel[3]) / stride[1] + 1;

    SpykerCompare(output[3], >, 0, "Shape::Conv", "Input and kernel shapes are not compatible.");
    SpykerCompare(output[4], >, 0, "Shape::Conv", "Input and kernel shapes are not compatible.");
    return output;
}

Shape fcShape(Shape input, Shape kernel)
{
    SpykerCompare(input.size(), ==, 3, "Shape::FC", "Input dimensions must be 3.");
    SpykerCompare(kernel.size(), ==, 2, "Shape::FC", "Kernel dimensions must be 2.");
    SpykerCompare(input[2], ==, kernel[1], "Shape::FC", "Input and kernel shapes don't match.");

    return Shape{input[0], input[1], kernel[0]};
}

Shape dogShape(Shape input, Shape kernel, Shape pad)
{
    SpykerCompare(input.size(), ==, 4, "Shape::DoG", "Input dimensions must be 4.");
    SpykerCompare(kernel.size(), ==, 4, "Shape::DoG", "Kernel dimensions must be 4.");
    SpykerCompare(pad.size(), ==, 4, "Shape::DoG", "Pad dimensions must be 4.");
    SpykerCompare(kernel[0], ==, 2, "Shape::DoG", "Kernel shape is invalid.");

    Shape output = {input[0], input[1] * kernel[1], 0, 0};
    output[2] = input[2] + pad[0] + pad[2] - kernel[2] + 1;
    output[3] = input[3] + pad[1] + pad[3] - kernel[3] + 1;

    SpykerCompare(output[2], >, 0, "Shape::DoG", "Input and kernel shapes are not compatible.");
    SpykerCompare(output[3], >, 0, "Shape::DoG", "Input and kernel shapes are not compatible.");
    return output;
}

Shape gaborShape(Shape input, Shape kernel, Shape pad)
{
    SpykerCompare(input.size(), ==, 4, "Shape::Gabor", "Input dimensions must be 4.");
    SpykerCompare(kernel.size(), ==, 3, "Shape::Gabor", "Kernel dimensions must be 3.");
    SpykerCompare(pad.size(), ==, 4, "Shape::Gabor", "Pad dimensions must be 4.");

    Shape output = {input[0], input[1] * kernel[0], 0, 0};
    output[2] = input[2] + pad[0] + pad[2] - kernel[1] + 1;
    output[3] = input[3] + pad[1] + pad[3] - kernel[2] + 1;

    SpykerCompare(output[2], >, 0, "Shape::Gabor", "Input and kernel shapes are not compatible.");
    SpykerCompare(output[3], >, 0, "Shape::Gabor", "Input and kernel shapes are not compatible.");
    return output;
}

Shape logShape(Shape input, Shape kernel, Shape pad)
{
    SpykerCompare(input.size(), ==, 4, "Shape::LoG", "Input dimensions must be 4.");
    SpykerCompare(kernel.size(), ==, 4, "Shape::LoG", "Kernel dimensions must be 4.");
    SpykerCompare(pad.size(), ==, 4, "Shape::LoG", "Pad dimensions must be 4.");
    SpykerCompare(kernel[0], ==, 2, "Shape::LoG", "Kernel shape is invalid.");

    Shape output = {input[0], input[1] * kernel[0] * kernel[1], 0, 0};
    output[2] = input[2] + pad[0] + pad[2] - kernel[2] + 1;
    output[3] = input[3] + pad[1] + pad[3] - kernel[3] + 1;

    SpykerCompare(output[2], >, 0, "Shape::LoG", "Input and kernel shapes are not compatible.");
    SpykerCompare(output[3], >, 0, "Shape::LoG", "Input and kernel shapes are not compatible.");
    return output;
}

Shape poolShape(Shape input, Shape kernel, Shape stride, Shape pad)
{
    SpykerCompare(input.size(), ==, 5, "Shape::Pool", "Input dimensions must be 5.");
    SpykerCompare(kernel.size(), ==, 2, "Shape::Pool", "Kernel dimensions must be 2.");
    SpykerCompare(stride.size(), ==, 2, "Shape::Pool", "Stride dimensions must be 2.");
    SpykerCompare(pad.size(), ==, 4, "Shape::Pool", "Pad dimensions must be 4.");

    Shape output = input;
    output[3] = (input[3] + pad[0] + pad[2] - kernel[0]) / stride[0] + 1;
    output[4] = (input[4] + pad[1] + pad[3] - kernel[1]) / stride[1] + 1;

    SpykerCompare(output[3], >, 0, "Shape::Pool", "Input and kernel shapes are not compatible.");
    SpykerCompare(output[4], >, 0, "Shape::Pool", "Input and kernel shapes are not compatible.");
    return output;
}

Shape padShape(Shape input, Shape pad)
{
    SpykerCompare(input.size(), ==, 5, "Shape::Pad", "Input dimensions must be 5.");
    SpykerCompare(pad.size(), ==, 4, "Shape::Pad", "Pad dimensions must be 4.");

    input[3] += pad[0] + pad[2], input[4] += pad[1] + pad[3];
    return input;
}

Shape codeShape(Shape input, Size time)
{
    SpykerCompare(input.size(), >, 1, "Shape::RankCode", "Input dimensions must be more than 1.");

    input.insert(input.begin() + 1, time);
    return input;
}

Shape gatherShape(Shape input)
{
    SpykerCompare(input.size(), >, 2, "Shape::Gather", "Input dimensions must more than 2.");
    input.erase(input.begin() + 1);
    return input;
}

Shape scatterShape(Shape input, Size time)
{
    SpykerCompare(input.size(), >, 1, "Shape::Scatter", "Input dimensions must be more than 1.");

    input.insert(input.begin() + 1, time);
    return input;
}

void zcaCheck(Shape input, Shape mean, Shape trans)
{
    SpykerCompare(input.size(), ==, 2, "Shape::ZCA", "Input dimensions must be 2.");
    SpykerCompare(mean.size(), ==, 1, "Shape::ZCA", "Mean dimensions must be 2.");
    SpykerCompare(trans.size(), ==, 2, "Shape::ZCA", "Transform dimensions must be 2.");
    SpykerCompare(input[1], ==, mean[0], "Shape::ZCA", "Input and mean shapes are not compatible.");
    SpykerCompare(input[1], ==, trans[0], "Shape::ZCA", "Input and mean shapes are not compatible.");
    SpykerCompare(input[1], ==, trans[1], "Shape::ZCA", "Input and mean shapes are not compatible.");
}

Shape zcaSplitShape(Shape input)
{
    SpykerCompare(input.size(), >, 1, "Shape::ZCA", "Input dimensions must be more than 1.");

    input.insert(input.begin() + 1, 2);
    return input;
}
}  // namespace Spyker
