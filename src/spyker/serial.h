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

# pragma once

#include <serio/serio.h>
#include <spyker/utils.h>

namespace Serio
{
template <>
struct CustomClass<Spyker::Tensor>
{
    template <typename Serializer>
    void serialize(const Spyker::Tensor &_data, Serializer &C)
    {
        using namespace Spyker;
        auto data = _data.cpu();

        C << I32(bool(data));
        if (!data) return;
        C << I32(data.type()) << data.shape();

        if (data.type() == Type::U8)
            C << Serio::Array<U8>(data.data<U8>(), data.numel());
        else if (data.type() == Type::U16)
            C << Serio::Array<U16>(data.data<U16>(), data.numel());
        else if (data.type() == Type::U32)
            C << Serio::Array<U32>(data.data<U32>(), data.numel());
        else if (data.type() == Type::U64)
            C << Serio::Array<U64>(data.data<U64>(), data.numel());

        if (data.type() == Type::I8)
            C << Serio::Array<I8>(data.data<I8>(), data.numel());
        else if (data.type() == Type::I16)
            C << Serio::Array<I16>(data.data<I16>(), data.numel());
        else if (data.type() == Type::I32)
            C << Serio::Array<I32>(data.data<I32>(), data.numel());
        else if (data.type() == Type::I64)
            C << Serio::Array<I64>(data.data<I64>(), data.numel());

        else if (data.type() == Type::F16)
            C << Serio::Array<U16>((U16 *)data.data<F16>(), data.numel());
        else if (data.type() == Type::F32)
            C << Serio::Array<F32>(data.data<F32>(), data.numel());
        else if (data.type() == Type::F64)
            C << Serio::Array<F64>(data.data<F64>(), data.numel());
        else
            SpykerAssert(false, "Serialization", "Unknown type given.");
    }
    template <typename Deserializer>
    void deserialize(Spyker::Tensor &data, Deserializer &C)
    {
        using namespace Spyker;

        I32 valid, type;
        C >> valid;
        if (!valid) return;

        Shape shape;
        C >> type >> shape;
        data = Tensor(Type(type), shape);

        if (data.type() == Type::U8)
        {
            Serio::Array<U8> array(data.data<U8>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::U16)
        {
            Serio::Array<U16> array(data.data<U16>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::U32)
        {
            Serio::Array<U32> array(data.data<U32>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::U64)
        {
            Serio::Array<U64> array(data.data<U64>(), data.numel());
            C >> array;
        }

        else if (data.type() == Type::I8)
        {
            Serio::Array<I8> array(data.data<I8>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::I16)
        {
            Serio::Array<I16> array(data.data<I16>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::I32)
        {
            Serio::Array<I32> array(data.data<I32>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::I64)
        {
            Serio::Array<I64> array(data.data<I64>(), data.numel());
            C >> array;
        }

        if (data.type() == Type::F16)
        {
            Serio::Array<U16> array((U16 *)data.data<F16>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::F32)
        {
            Serio::Array<F32> array(data.data<F32>(), data.numel());
            C >> array;
        }
        else if (data.type() == Type::F64)
        {
            Serio::Array<F64> array(data.data<F64>(), data.numel());
            C >> array;
        }
        else
            SpykerAssert(false, "Serialization", "Unknown type given.");
    }
};
}  // namespace Serio
