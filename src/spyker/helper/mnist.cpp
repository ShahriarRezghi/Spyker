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
