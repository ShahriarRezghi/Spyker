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
