#include "serial.h"

#include <serio/serio.h>
#include <spyker/base.h>

#include <fstream>

namespace Spyker
{
namespace Helper
{
struct SerialTensor
{
    Tensor data;

    template <typename Serializer>
    void _serialize(Serializer &C) const
    {
        C << I32(bool(data));
        if (!data) return;
        C << I32(data.type()) << data.shape();

        if (data.type() == Type::F16)
        {
            C << Serio::Array<U16>((U16 *)data.data<F16>(), data.numel());
        }
        else
        {
            IfNotHalf(T, data.type(), C << Serio::Array<T>(data.data<T>(), data.numel()));
        }
    }
    template <typename Deserializer>
    void _deserialize(Deserializer &C)
    {
        I32 valid, type;
        C >> valid;
        if (!valid) return;

        Shape shape;
        C >> type >> shape;
        data = Tensor(Type(type), shape);

        if (data.type() == Type::F16)
        {
            Serio::Array<U16> array((U16 *)data.data<F16>(), data.numel());
            C >> array;
        }
        else
        {
            IfNotHalf(T, data.type(), Serio::Array<T> array(data.data<T>(), data.numel()); C >> array);
        }
    }
};

#define VERSION 1

void serialize(std::ofstream &file, Tensor tensor)
{
    tensor = tensor.to(Kind::CPU);
    Serio::write(&file, *(SerialTensor *)&tensor);
}

bool serialize(const std::string &path, const std::vector<Tensor> &list)
{
    std::ofstream file(path, std::ios::out | std::ios::binary);
    Serio::write(&file, uint32_t(VERSION), Serio::Size(list.size()));
    if (file.fail()) return false;

    for (const auto &tensor : list)
    {
        serialize(file, tensor);
        if (file.fail()) return false;
    }
    return true;
}

std::vector<Tensor> deserialize(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    uint32_t version;
    Serio::read(&file, version);
    SpykerCompare(version, <=, VERSION, "Helper::Serial", "Input file has higher version than supported.");

    std::vector<Tensor> list;
    Serio::read(&file, (std::vector<SerialTensor> &)list);
    return file.fail() ? std::vector<Tensor>() : list;
}
}  // namespace Helper
}  // namespace Spyker
