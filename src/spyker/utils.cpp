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

#include "utils.h"

#include "base.h"
#include "impl.h"

#define ValidCheck \
    SpykerCompare(bool(_data), ==, true, "Interface::Tensor", "Tensor can't be used without initialization.");

namespace Spyker
{
Size numel(const Shape &shape)
{
    Size size = 1;
    for (Size item : shape) size *= item;
    return size;
}

void check(const Shape &shape)
{
    SpykerCompare(shape.size(), >, 0, "Interface::Tensor", "Tensor Shape can't be empty.");
    for (auto item : shape)
        SpykerCompare(item, >, 0, "Interface::Tensor", "Tensor Shape elements can't be less than 1.");
}

Tensor::Tensor(Shared data, Size offset, Device device, bool pinned, bool unified, Type type, Shape shape)
    : _data(data), _offset(offset), _device(device), _pinned(pinned), _unified(unified), _type(type), _shape(shape)
{
}

Tensor::Tensor() : _offset(0), _type(Type::F32), _device(Kind::CPU) {}

Tensor::Tensor(Type type, const Shape &shape, bool pinned, bool unified)  //
    : Tensor(Kind::CPU, type, shape, pinned, unified)
{
}

Tensor::Tensor(Shared data, Type type, const Shape &shape, bool pinned, bool unified)  //
    : Tensor(data, Kind::CPU, type, shape, pinned, unified)
{
}

Tensor::Tensor(Device device, Type type, const Shape &shape, bool pinned, bool unified)
    : _offset(0), _device(device), _type(type), _shape(shape), _pinned(pinned), _unified(unified)
{
    check(shape);

    if (_unified)
    {
        SpykerAssert(!_pinned, "Interface::Tensor", "Unified tensor can't be pinned.");
        auto data = Core::unified_alloc(Spyker::numel(shape) * TypeSize(type));
        _data = Shared(data, [](void *data) { Core::unified_dealloc(data); });
    }
    else if (_pinned)
    {
        SpykerCompare(device, ==, Kind::CPU, "Interface::Tensor", "Tensor with pinned memory must be on CPU.");
        auto data = Core::pinned_alloc(Spyker::numel(shape) * TypeSize(type));
        _data = Shared(data, [](void *data) { Core::pinned_dealloc(data); });
    }
    else
    {
        auto data = Core::alloc(Spyker::numel(shape) * TypeSize(type), device);
        _data = Shared(data, [device](void *data) { Core::dealloc(data, device); });
    }
}

Tensor::Tensor(Shared data, Device device, Type type, const Shape &shape, bool pinned, bool unified)
    : _data(data), _offset(0), _device(device), _type(type), _shape(shape), _pinned(pinned), _unified(unified)
{
    SpykerAssert(!(pinned && unified), "Interface::Tensor", "Tensor can't be both pinned and unified.") check(shape);
}

bool Tensor::pinned() const { ValidCheck return _pinned; }

bool Tensor::unified() const { ValidCheck return _unified; }

Tensor::operator bool() const { return bool(_data); }

void *Tensor::data() const { return _data ? (I8 *)_data.get() + _offset : nullptr; }

Size Tensor::dims() const { return _data ? _shape.size() : 0; }

Size Tensor::numel() const { return _data ? Spyker::numel(_shape) : 0; }

Shape Tensor::shape() const { return _data ? _shape : Shape(); }

Size Tensor::bytes() const { ValidCheck return numel() * TypeSize(_type); }

Size Tensor::shape(Size index) const { ValidCheck return _shape.at(index); }

Type Tensor::type() const { ValidCheck return _type; }

Device Tensor::device() const { ValidCheck return _device; }

Tensor Tensor::copy() const
{
    if (!_data) return *this;
    Tensor output(_device, _type, _shape);
    Core::copy(bytes(), data(), output.data(), _device);
    return output;
}

Tensor Tensor::to(Type type) const
{
    ValidCheck if (_type == type) return *this;
    Tensor output(_device, type, _shape);
    Core::cast(numel(), Dyn(data(), _type), Dyn(output.data(), type), _device);
    return output;
}

Tensor Tensor::to(Device device) const
{
    ValidCheck if (_device == device) return *this;
    if (_unified) return Tensor(_data, device, _type, _shape, _pinned, _unified);
    Tensor output(device, _type, _shape);
    Core::transfer(bytes(), data(), _device, output.data(), device);
    return output;
}

void Tensor::to(Tensor other) const
{
    ValidCheck SpykerAssert(other, "Interface::Tensor", "Tensors must be initialized before copying.");
    SpykerCompare(numel(), ==, other.numel(), "Interface::Tensor", "Tensors must have the same numel.");

    auto input = *this;

    if (_device != other._device)
    {
        if (_type != other._type) input = input.to(other._type);
        Core::transfer(input.bytes(), input.data(), input._device, other.data(), other._device);
        return;
    }
    if (_type != other._type)
    {
        if (_device != other._device) input = input.to(other._device);
        Core::cast(numel(), Dyn(input.data(), input._type), Dyn(other.data(), other._type), input._device);
        return;
    }
    Core::copy(bytes(), data(), other.data(), _device);
}

void Tensor::from(Tensor tensor) const { ValidCheck tensor.to(*this); }

Tensor Tensor::reshape(Shape shape) const
{
    auto size = Spyker::numel(_shape);
    auto it = std::find(shape.begin(), shape.end(), -1);
    if (it != shape.end()) *it = 1, *it = size / Spyker::numel(shape);
    SpykerAssert(std::find(shape.begin(), shape.end(), -1) == shape.end(), "Interface::Tensor",
                 "Given shape contains more than one placeholder (-1)");

    ValidCheck check(shape);
    SpykerCompare(Spyker::numel(shape), ==, Spyker::numel(_shape), "Interface::Tensor",
                  "New and old shape must have the same numel.");
    return Tensor(_data, _offset, _device, _pinned, _unified, _type, shape);
}

Tensor &Tensor::fill(Scalar value)
{
    ValidCheck Core::fill(numel(), Dyn(data(), _type), value.to(_type), _device);
    return *this;
}

Tensor Tensor::operator[](Size index) const
{
    ValidCheck;
    SpykerCompare(index, <, _shape[0], "Interface::Tensor", "Given index is out of range.");
    Shape shape(_shape.begin() + 1, _shape.end());
    Size offset = _offset + index * Spyker::numel(shape) * TypeSize(_type);
    return Tensor(_data, offset, _device, _pinned, _unified, _type, shape);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define GET *(Sparse5 *)

#undef ValidCheck

#define ValidCheck \
    SpykerCompare(bool(_data), ==, true, "Interface::Sparse", "Sparse can't be used without initialization.");

SparseTensor::SparseTensor(Shape shape) : _shape(shape)
{
    auto deleter = [](void *data) {
        Core::sparse_dealloc(GET data);
        delete (Sparse5 *)data;
    };
    _data = Shared(new Sparse5(Core::sparse_alloc(shape)), deleter);
}

SparseTensor::SparseTensor(Tensor tensor, F64 threshold) : SparseTensor(tensor.shape())
{
    Core::sparse_convert(tensor, GET data(), threshold);
}

SparseTensor::operator bool() const { return bool(_data); }

Size SparseTensor::dims() const { return _data ? _shape.size() : 0; }

Size SparseTensor::numel() const { return Core::sparse_elemsize(GET data()); }

void *SparseTensor::data() const { return _data ? _data.get() : nullptr; }

Shape SparseTensor::shape() const { return _data ? _shape : Shape(); }

Size SparseTensor::shape(Size index) const { ValidCheck return _shape.at(index); }

Size SparseTensor::bytes() const { ValidCheck return Core::sparse_memsize(GET data()); }

F64 SparseTensor::sparsity() const { ValidCheck return F64(numel()) / Spyker::numel(_shape); }

SparseTensor SparseTensor::copy() const
{
    if (!_data) return *this;
    SparseTensor output(_shape);
    Core::sparse_copy(GET data(), GET output.data());
    return output;
}

Tensor SparseTensor::dense() const
{
    ValidCheck;
    Tensor output(Type::U8, _shape);
    Core::sparse_convert(GET data(), output);
    return output;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Size TypeSize(Type type) { IfType(T, type, return sizeof(T)); }

Device::Device(Kind kind, I32 index) : _kind(kind), _index(index)
{
    if (_kind == Kind::CPU)
    {
        if (_index < 0) _index = 0;
        SpykerCompare(_index, ==, 0, "Interface::Device", "Only index 0 is supported for CPU device.");
    }
    else if (_kind == Kind::CUDA)
    {
        if (_index < 0) _index = Core::cuda_current_device();
        SpykerCompare(_index, <, Core::cuda_device_count(), "Interface::Device",
                      "Given CUDA device index doesn't exist.");
    }
    else
        SpykerAssert(false, "Interface::Device", "Given device is not recognized.");
}

bool Device::operator==(Kind kind) const { return _kind == kind; }

bool Device::operator!=(Kind kind) const { return _kind != kind; }

bool Device::operator==(Device other) const { return _kind == other._kind && _index == other._index; }

bool Device::operator!=(Device other) const { return _kind != other._kind || _index != other._index; }

Scalar Scalar::to(Type type) const
{
    SpykerCompare(_valid, ==, true, "Interface::Scalar", "Scalar is used before it is initialized.");
    if (type == _type) return *this;
    IfType(O, type, IfType(I, _type, return O(value<I>())));
}

std::ostream &operator<<(std::ostream &os, Type type)
{
    if (type == Type::I8) return os << "I8";
    if (type == Type::I16) return os << "I16";
    if (type == Type::I32) return os << "I32";
    if (type == Type::I64) return os << "I64";
    if (type == Type::U8) return os << "U8";
    if (type == Type::U16) return os << "U16";
    if (type == Type::U32) return os << "U32";
    if (type == Type::U64) return os << "U64";
    if (type == Type::F16) return os << "F16";
    if (type == Type::F32) return os << "F32";
    if (type == Type::F64) return os << "F64";
    return os << "Unknown";
}

std::ostream &operator<<(std::ostream &os, Kind kind)
{
    if (kind == Kind::CPU) return os << "CPU";
    if (kind == Kind::CUDA) return os << "CUDA";
    return os << "Unknown";
}

std::ostream &operator<<(std::ostream &os, Device device)  //
{
    return os << device.kind() << ":" << device.index();
}

std::ostream &operator<<(std::ostream &os, Shape shape)
{
    os << "<";
    for (size_t i = 0; i < shape.size(); ++i)  //
        os << shape[i] << (i + 1 == shape.size() ? "" : "x");
    return os << ">";
}

std::ostream &operator<<(std::ostream &os, Scalar scalar)
{
    if (!scalar) return os << "Scalar<>";
    os << "Scalar<" << scalar.type() << ", ";
    IfType(T, scalar.type(), os << F64(scalar.value<T>()));
    return os << ">";
}

template <typename T>
void printTensor(Shape shape, T *data, std::ostream &os)
{
    if (shape.size() == 1)
        shape = {1, 1, shape[0]};
    else if (shape.size() == 2)
        shape = {1, shape[0], shape[1]};
    else if (shape.size() > 3)
        shape = {numel(Shape(shape.begin(), shape.end() - 2)), shape[shape.size() - 2], shape[shape.size() - 1]};

    for (Size i = 0; i < shape[0]; ++i)
    {
        os << i + 1 << ":" << std::endl;
        for (Size j = 0; j < shape[1]; ++j)
        {
            os << "    ";
            auto dd = data + (i * shape[1] + j) * shape[2];
            for (Size k = 0; k < shape[2]; ++k) os << F64(dd[k]) << (k != shape[2] - 1 ? ", " : "");
            os << std::endl;
        }
        if (i != shape[0] - 1) os << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, Tensor tensor)
{
    if (!tensor) return os << "Tensor<>";
    os << "Tensor<";
    auto shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i)  //
        os << shape[i] << (i + 1 == shape.size() ? "" : "x");
    os << ", " << tensor.type() << ", " << tensor.device() << ">" << std::endl;

    tensor = tensor.cpu();
    IfType(T, tensor.type(), printTensor(shape, tensor.data<T>(), os));
    return os;
}
}  // namespace Spyker
