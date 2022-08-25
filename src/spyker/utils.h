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

#include <spyker/half.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define SpykerExport __declspec(dllexport)
#else
#define SpykerExport
#endif

#define SpykerAssert(expr, scope, message)                                                                            \
    if (!static_cast<bool>(expr))                                                                                     \
    {                                                                                                                 \
        std::ostringstream stream("");                                                                                \
        stream << "Failure At: " << __FILE__ << ":" << __LINE__ << ", Expression: \"" #expr << "\", Scope: " << scope \
               << ", Message: " << message;                                                                           \
        throw std::runtime_error(stream.str());                                                                       \
    }

#define SpykerCompare(first, op, second, scope, message)                                                            \
    if (!static_cast<bool>(first op second))                                                                        \
    {                                                                                                               \
        std::ostringstream stream("");                                                                              \
        stream << "Failure At: " << __FILE__ << ":" << __LINE__ << ", Expression: \"" << first << " " << #op << " " \
               << second << "\", Scope: " << scope << ", Message: " << message;                                     \
        throw std::runtime_error(stream.str());                                                                     \
    }

namespace Spyker
{
/// 8-bit signed integer data type.
using I8 = int8_t;
/// 16-bit signed integer data type.
using I16 = int16_t;
/// 32-bit signed integer data type.
using I32 = int32_t;
/// 64-bit signed integer data type.
using I64 = int64_t;
/// 8-bit unsigned integer data type.
using U8 = uint8_t;
/// 16-bit unsigned integer data type.
using U16 = uint16_t;
/// 32-bit unsigned integer data type.
using U32 = uint32_t;
/// 64-bit unsigned integer data type.
using U64 = uint64_t;
/// half percision floating point data type
using F16 = half_float::half;
/// single percision floating point data type.
using F32 = float;
/// double percision floating point data type.
using F64 = double;
/// Size type of the library.
using Size = int64_t;

/// Data type enumeration of the library.
enum class Type : I32
{
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// half percision floating point.
    F16,
    /// single percision floating point.
    F32,
    /// double percision floating point.
    F64,
};

enum class Code : I32
{
    /// Rank coding
    Rank,
    /// Rate coding
    Rate,
};

/// Get the name of the templated data type.
template <typename T>
Type TypeName()
{
    if (std::is_same<T, I8>()) return Type::I8;
    if (std::is_same<T, I16>()) return Type::I16;
    if (std::is_same<T, I32>()) return Type::I32;
    if (std::is_same<T, I64>()) return Type::I64;
    if (std::is_same<T, U8>()) return Type::U8;
    if (std::is_same<T, U16>()) return Type::U16;
    if (std::is_same<T, U32>()) return Type::U32;
    if (std::is_same<T, U64>()) return Type::U64;
    if (std::is_same<T, F16>()) return Type::F16;
    if (std::is_same<T, F32>()) return Type::F32;
    if (std::is_same<T, F64>()) return Type::F64;
    SpykerAssert(false, "Interface::Type", "Given type is not recognized.");
}

/// Get the size of the type in bytes.
Size SpykerExport TypeSize(Type type);

/// Kind of the device.
enum class Kind : I32
{
    /// CPU device.
    CPU,
    /// CUDA GPU device.
    CUDA,
};

/// Device to use for allocations and computations.
class SpykerExport Device
{
    /// Kind of the device.
    Kind _kind;
    /// Index of the device.
    I32 _index;

public:
    /// @param kind kind of the device.
    /// @param index index of the device.
    Device(Kind kind = Kind::CPU, I32 index = -1);

    /// Get the kind of the device.
    inline Kind kind() const { return _kind; }

    /// Get the index of the device.
    inline I32 index() const { return _index; }

    /// Equality operator of the device.
    /// @param kind kind of the device to be compared to.
    bool operator==(Kind kind) const;

    /// Non-equality operator of the device.
    /// @param kind kind of the device to be compared to.
    bool operator!=(Kind kind) const;

    /// Equality operator of the device.
    /// @param other device to be compared to.
    bool operator==(Device other) const;

    /// Non-equality operator of the device.
    /// @param other device to be compared to.
    bool operator!=(Device other) const;
};

class Scalar;

/// Shape container of the library.
using Shape = std::vector<Size>;

/// N-dimensional container of the data buffers on different devices.
class SpykerExport Tensor
{
public:
    using Shared = std::shared_ptr<void>;

private:
    Shared _data;
    Size _offset;
    Device _device;
    bool _pinned = false;
    bool _unified = false;
    Type _type;
    Shape _shape;

    Tensor(Shared data, Size offset, Device device, bool pinned, bool unified, Type type, Shape shape);

public:
    /// Data buffer deleter function.
    using Deleter = std::function<void(void *)>;

    /// Initialize empty tensor.
    Tensor();

    /// @param type type of the tensor.
    /// @param shape shape of the tensor.
    /// @param pinned whether the tensor is pinned (CUDA pinned memory) or not.
    /// @param unified whether the tensor is unified (CUDA unified memory) or not.
    Tensor(Type type, const Shape &shape, bool pinned = false, bool unified = false);

    /// @param data user raw data to be held.
    /// @param type type of the tensor.
    /// @param shape shape of the tensor.
    /// @param pinned whether the tensor is pinned (CUDA pinned memory) or not.
    /// @param unified whether the tensor is unified (CUDA unified memory) or not.
    Tensor(Shared data, Type type, const Shape &shape, bool pinned = false, bool unified = false);

    /// @param device device of the tensor.
    /// @param type type of the tensor.
    /// @param shape shape of the tensor.
    /// @param pinned whether the tensor is pinned (CUDA pinned memory) or not.
    /// @param unified whether the tensor is unified (CUDA unified memory) or not.
    Tensor(Device device, Type type, const Shape &shape, bool pinned = false, bool unified = false);

    /// @param data user raw data to be held.
    /// @param device device of the tensor.
    /// @param type type of the tensor.
    /// @param shape shape of the tensor.
    /// @param pinned whether the tensor is pinned (CUDA pinned memory) or not.
    /// @param unified whether the tensor is unified (CUDA unified memory) or not.
    Tensor(Shared data, Device device, Type type, const Shape &shape, bool pinned = false, bool unified = false);

    /// Check whether the tensor is pinned (CUDA pinned memory).
    bool pinned() const;

    /// Check whether the tensor is unified (CUDA unified memory).
    bool unified() const;

    /// Check whether the tensor is valid (not empty).
    operator bool() const;

    /// Get the raw data pointer.
    void *data() const;

    /// Get the number of dimensions.
    Size dims() const;

    /// Get the number of elements in tensor.
    Size numel() const;

    /// Get the sie of data buffer in bytes.
    Size bytes() const;

    /// Get the shape of the tensor.
    Shape shape() const;

    /// Get the shape element in index.
    /// @param index index of the element to be returned.
    Size shape(Size index) const;

    /// Get the data type of the tensor.
    Type type() const;

    /// Get the device of the tensor.
    Device device() const;

    /// Get a deep copy of the tensor.
    Tensor copy() const;

    /// Return a copy of the tensor with converted data type.
    Tensor to(Type type) const;

    /// Return a copy of the tensor with converted device.
    Tensor to(Device device) const;

    /// Copy the tensor into the other.
    /// @param other tensor to be copied into.
    void to(Tensor other) const;

    /// Copy the other tensor into this.
    /// @param tensor tensor to copy from.
    void from(Tensor tensor) const;

    /// Reshape the tensor.
    /// @param shape new shape of the tensor.
    Tensor reshape(Shape shape) const;

    /// Fill the tensor with value.
    /// @param value value to fill the tensor with.
    Tensor &fill(Scalar value);

    /// Get subtensor.
    Tensor operator[](Size index) const;

    /// Get the templated data pointer.
    template <typename T>
    T *data() const
    {
        SpykerCompare(TypeName<T>(), ==, _type, "Interface::Tensor",
                      "Requested pointer type is not the same as tensor's.");
        return (T *)data();
    }

    /// Return a copy of this tensor with data type converted to 8-bit signed integer.
    inline Tensor i8() const { return to(Type::I8); }
    /// Return a copy of this tensor with data type converted to 16-bit signed integer.
    inline Tensor i16() const { return to(Type::I16); }
    /// Return a copy of this tensor with data type converted to 32-bit signed integer.
    inline Tensor i32() const { return to(Type::I32); }
    /// Return a copy of this tensor with data type converted to 64-bit signed integer.
    inline Tensor i64() const { return to(Type::I64); }
    /// Return a copy of this tensor with data type converted to 8-bit unsigned integer.
    inline Tensor u8() const { return to(Type::U8); }
    /// Return a copy of this tensor with data type converted to 16-bit unsigned integer.
    inline Tensor u16() const { return to(Type::U16); }
    /// Return a copy of this tensor with data type converted to 32-bit unsigned integer.
    inline Tensor u32() const { return to(Type::U32); }
    /// Return a copy of this tensor with data type converted to 64-bit unsigned integer.
    inline Tensor u64() const { return to(Type::U64); }
    /// Return a copy of this tensor with data type converted to half percision floating point.
    inline Tensor f16() const { return to(Type::F16); }
    /// Return a copy of this tensor with data type converted to single percision floating point.
    inline Tensor f32() const { return to(Type::F32); }
    /// Return a copy of this tensor with data type converted to double percision floating point.
    inline Tensor f64() const { return to(Type::F64); }
    /// Return a copy of this tensor with device converted to CPU.
    inline Tensor cpu() const { return to(Kind::CPU); }
    /// Return a copy of this tensor with device converted to CUDA.
    inline Tensor cuda() const { return to(Kind::CUDA); }

    template <typename T>
    static Tensor hold(T *data, const Shape &shape, bool pinned = false, bool unified = false)
    {
        return Tensor(Shared(data, [](void *) {}), Kind::CPU, TypeName<T>(), shape, pinned, unified);
    }
    template <typename T>
    static Tensor hold(T *data, Device device, const Shape &shape, bool pinned = false, bool unified = false)
    {
        return Tensor(Shared(data, [](void *) {}), device, TypeName<T>(), shape, pinned, unified);
    }
    template <typename T>
    static Tensor hold(std::vector<T> &data, const Shape &shape, bool pinned = false, bool unified = false)
    {
        return Tensor(Shared(data.data(), [](void *) {}), Kind::CPU, TypeName<T>(), shape, pinned, unified);
    }
};

/// 5-dimentional container of sparse binary spikes on the CPU
class SpykerExport SparseTensor
{
    using Shared = std::shared_ptr<void>;

    Shared _data;
    Shape _shape;

public:
    /// Initialize empty tensor
    SparseTensor();

    /// @param shape shape of the tensor
    SparseTensor(Shape shape);

    /// Initialize from a dense tensor
    /// @param tensor dense tensor
    /// @param threshold the threshold used to turn dense tensor into spikes
    explicit SparseTensor(Tensor tensor, F64 threshold = 0);

    /// Check whether the tensor is valid (not empty).
    operator bool() const;

    /// Get the number of dimensions.
    Size dims() const;

    /// Get the number of elements in tensor.
    Size numel() const;

    /// Get the raw data pointer. This points to a internal data structure and should not be used.
    void *data() const;

    /// Get the shape of the tensor.
    Shape shape() const;

    /// Get the shape element in index.
    /// @param index index of the element to be returned.
    Size shape(Size index) const;

    /// Get the sie of data buffer in bytes.
    Size bytes() const;

    /// Get the ratio of non-zero elements in the range of zero to one.
    F64 sparsity() const;

    /// Get a deep copy of the tensor.
    SparseTensor copy() const;

    /// Convert the sparse tensor into a dense tensor.
    Tensor dense() const;
};

SpykerExport std::ostream &operator<<(std::ostream &os, Type type);
SpykerExport std::ostream &operator<<(std::ostream &os, Kind kind);
SpykerExport std::ostream &operator<<(std::ostream &os, Device device);
SpykerExport std::ostream &operator<<(std::ostream &os, Shape shape);
SpykerExport std::ostream &operator<<(std::ostream &os, Tensor tensor);
SpykerExport std::ostream &operator<<(std::ostream &os, Scalar scalar);

/// Non-templated basic data type container.
class SpykerExport Scalar
{
    bool _valid;
    Type _type;
    I64 _data;

public:
    /// Initialize empty scalar
    inline Scalar() : _valid(false) {}

    /// Get the data type of the scalar.
    inline Type type() const { return _type; }

    /// Check if the scalar is valid (not empty).
    inline operator bool() const { return _valid; }

    /// @param value value to be assigned to the scalar.
    template <typename T>
    Scalar(T value) : _valid(true), _type(TypeName<T>())
    {
        *(T *)&_data = value;
    }

    /// Assign a new value to the scalar.
    template <typename T>
    Scalar &operator=(T value)
    {
        _valid = true;
        _type = TypeName<T>();
        *(T *)&_data = value;
        return *this;
    }

    /// Return the templated data type.
    template <typename T>
    T value() const
    {
        SpykerCompare(_valid, ==, true, "Interface::Scalar", "Scalar is used before it is initialized.");
        SpykerCompare(_type, ==, TypeName<T>(), "Interface::Scalar",
                      "Requested data type is not the same as scalar's.");
        return *(T *)&_data;
    }

    /// Cast operator to basic types.
    template <typename T>
    operator T() const
    {
        return to(TypeName<T>()).template value<T>();
    }

    /// Get a copy of the scalar with the requested data type.
    Scalar to(Type type) const;
};

/// Winner neuron representation.
struct Winner
{
    /// Index of the STDP configuration.
    Size c;

    /// Index in time (used in sparse).
    Size t;

    /// Index in signal channel.
    Size z;

    /// Index in signal height.
    Size y;

    /// Index in signal width.
    Size x;
};

/// List of list of 'Winner's.
using Winners = std::vector<std::vector<Winner>>;

/// STDP configuration parameters container.
struct SpykerExport STDPConfig
{
    /// Stabilization.
    bool stabilize;
    /// Positive learning rate.
    F64 pos;
    /// Negative learning rate.
    F64 neg;
    /// Lower bound of the weights.
    F64 low;
    /// Upper bound of the weights.
    F64 high;

    inline STDPConfig() {}

    inline STDPConfig(F64 pos, F64 neg, bool stabilize = true, F64 low = 0, F64 high = 1)
        : stabilize(stabilize), pos(pos), neg(neg), low(low), high(high)
    {
    }
};

/// Backpropagation configuration parameters container.
struct SpykerExport BPConfig
{
    /// Scaling factor
    F32 sfactor;

    /// Learning rate
    F32 lrate;

    /// Learning rate of scaling factor
    F32 lrf;

    /// Weight regularization term
    F32 lambda;

    inline BPConfig() : sfactor(1), lrate(1), lrf(1), lambda(0) {}

    inline BPConfig(F32 sfactor, F32 lrate, F32 lrf, F32 lambda)
        : sfactor(sfactor), lrate(lrate), lrf(lrf), lambda(lambda)
    {
    }
};
}  // namespace Spyker
