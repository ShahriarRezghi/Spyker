#include "image.h"

#include <spyker/impl.h>

#include <cstring>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>

namespace Spyker
{
namespace Helper
{
int readMode(const std::string &mode)
{
    if (mode == "") return 0;
    if (mode == "Y") return 1;
    if (mode == "YA") return 2;
    if (mode == "RGB") return 3;
    if (mode == "RGBA") return 4;
    SpykerAssert(false, "Helper::Image",
                 "Incorrect mode for reading image. Options are \"\", \"Y\", \"YA\", \"RGB\", \"RGBA\".");
}

Tensor readHWC(const std::string &path, std::string _mode)
{
    for (auto &c : _mode) c = toupper(c);
    int H = 0, W = 0, C = 0;
    auto mode = readMode(_mode);
    auto ptr = stbi_load(path.c_str(), &W, &H, &C, mode);
    if (ptr == nullptr) return Tensor();
    if (mode != 0) C = mode;
    std::shared_ptr<void> data(ptr, [](void *data) { std::free(data); });
    return Tensor(data, Kind::CPU, Type::U8, {H, W, C});
}

bool writeHWC(Tensor data, const std::string &path, std::string format)
{
    for (auto &c : format) c = toupper(c);
    SpykerCompare(data.dims(), ==, 3, "Helper::Image", "Input tensor must be 3 dimensional.");
    int H = data.shape(0), W = data.shape(1), C = data.shape(2);
    if (format == "PNG") return stbi_write_png(path.c_str(), W, H, C, data.data(), 0) != 0;
    if (format == "BMP") return stbi_write_bmp(path.c_str(), W, H, C, data.data()) != 0;
    if (format == "TGA") return stbi_write_tga(path.c_str(), W, H, C, data.data()) != 0;
    if (format == "JPG") return stbi_write_jpg(path.c_str(), W, H, C, data.data(), 100) != 0;
    SpykerAssert(false, "Helper::Image",
                 "Incorrect mode for writing image. Options are \"PNG\", \"BMP\", \"TGA\", \"JPG\".");
}

bool resizeHWC(Tensor input, Tensor output)
{
    return stbir_resize_uint8(input.data<U8>(), input.shape(0), input.shape(1), 0,  //
                              output.data<U8>(), output.shape(0), output.shape(1), 0, input.shape(2)) == 1;
}

bool readImage(Tensor output, const std::string &path, std::string mode, Shape size)
{
    SpykerCompare(size.size(), ==, 2, "Helper::Image", "Input size must have 2 dimensions.");
    SpykerCompare(output.dims(), ==, 3, "Helper::Image", "Output tensor must be 3 dimensional.");

    auto input = readHWC(path, mode);
    if (!input) return false;

    if (size[0] != -1 && size[1] != -1)
    {
        Tensor middle(Type::U8, {size[0], size[1], input.shape(2)});
        if (!resizeHWC(input, middle)) return false;
        input = middle;
    }

    Core::hwc2chw(output.shape(), input.data<U8>(), output.data<U8>());
    return true;
}

Tensor readImage(const std::string &path, std::string mode, Shape size)
{
    SpykerCompare(size.size(), ==, 2, "Helper::Image", "Input size must have 2 dimensions.");

    auto input = readHWC(path, mode);
    if (!input) return Tensor();

    if (size[0] != -1 && size[1] != -1)
    {
        Tensor middle(Type::U8, {size[0], size[1], input.shape(2)});
        if (!resizeHWC(input, middle)) return Tensor();
        input = middle;
    }

    Tensor output(Type::U8, {input.shape(2), input.shape(0), input.shape(1)});
    Core::hwc2chw(output.shape(), input.data<U8>(), output.data<U8>());
    return output;
}

bool writeImage(Tensor input, const std::string &path, std::string format)
{
    SpykerCompare(input.dims(), ==, 3, "Helper::Image", "Input tensor must be 3 dimensional.");
    Tensor output(Type::U8, {input.shape(1), input.shape(2), input.shape(0)});
    Core::chw2hwc(input.shape(), input.data<U8>(), output.data<U8>());
    return writeHWC(output, path, format);
}
}  // namespace Helper
}  // namespace Spyker
