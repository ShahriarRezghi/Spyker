#include "opers.h"

#include <omp.h>

#include <cmath>
#include <thread>

#include "impl.h"
#include "shape.h"

#define INIT(name)                 \
    name::name() : _init(false) {} \
    Device name::device() const { return _device; }

namespace Spyker
{
void randomSeed(Size seed) { return Core::random_seed(seed); }

bool cudaAvailable() { return Core::cuda_available(); }

Size cudaDeviceCount() { return Core::cuda_device_count(); }

void cudaSetDevice(Size index) { Core::cuda_set_device(index); }

Size cudaCurrentDevice() { return Core::cuda_current_device(); }

std::vector<Size> cudaArchList() { return Core::cuda_arch_list(); }

std::vector<Size> cudaDeviceArch() { return Core::cuda_device_arch(); }

Size cudaMemoryTotal() { return Core::cuda_memory_total(); }

Size cudaMemoryFree() { return Core::cuda_memory_free(); }

Size cudaMemoryTaken(Size device)
{
    if (device < 0) device = Core::cuda_current_device();
    return Core::cuda_memory_taken(device);
}

Size cudaMemoryUsed(Size device)
{
    if (device < 0) device = Core::cuda_current_device();
    return Core::cuda_memory_used(device);
}

bool cudaCacheEnabled(Size device)
{
    if (device < 0) device = Core::cuda_current_device();
    return Core::cuda_cache_enabled(device);
}

void cudaCacheEnable(bool enable, Size device)
{
    if (device < 0) device = Core::cuda_current_device();
    return Core::cuda_cache_enable(enable, device);
}

void cudaCacheClear(Size device)
{
    if (device < 0) device = Core::cuda_current_device();
    return Core::cuda_cache_clear(device);
}

void clearContext()
{
    Core::cpu_conv_clear();
    Core::cpu_fc_clear();
    Core::cpu_poisson_clear();

    if (cudaAvailable())
    {
        Core::cuda_conv_clear();
        Core::cuda_poisson_clear();
        cudaCacheClear();
    }
}

void cudaCachePrint(Size device)
{
    if (device < 0) device = Core::cuda_current_device();
    Core::cuda_cache_print(device);
}

void lightConv(bool light)
{
    if (cudaAvailable()) Core::cuda_light_conv(light);
}

std::vector<Device> allDevices() { return Core::devices(); }

Size maxThreads() { return omp_get_max_threads(); }

void setMaxThreads(Size threads)
{
    if (threads < 0) threads = std::thread::hardware_concurrency();
    omp_set_num_threads(threads);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void check(const char *name, Expand2 A, Expand2 B)
{
    SpykerCompare(A[0], >=, B[0], "Interface", name << " items are incorrect.");
    SpykerCompare(A[1], >=, B[1], "Interface", name << " items are incorrect.");
}
void check(const char *name, Expand4 A, Expand4 B)
{
    SpykerCompare(A[0], >=, B[0], "Interface", name << " items are incorrect.");
    SpykerCompare(A[1], >=, B[1], "Interface", name << " items are incorrect.");
    SpykerCompare(A[2], >=, B[2], "Interface", name << " items are incorrect.");
    SpykerCompare(A[3], >=, B[3], "Interface", name << " items are incorrect.");
}

Tensor least2(Tensor input)
{
    auto shape = input.shape();
    if (shape.size() == 1) shape.insert(shape.begin(), 1);
    SpykerCompare(shape.size(), >, 1, "Interface", "Input dimensions couldn't be viewed as at least 2D.");
    return input.reshape(shape);
}
Tensor least3(Tensor input)
{
    auto shape = input.shape();
    if (shape.size() == 2) shape.insert(shape.begin(), 1);
    SpykerCompare(shape.size(), >, 2, "Interface", "Input dimensions couldn't be viewed as at least 3D.");
    return input.reshape(shape);
}
Tensor to2(Tensor input)
{
    auto shape = input.shape();
    if (shape.size() == 1) shape.insert(shape.begin(), 1);
    SpykerCompare(shape.size(), ==, 2, "Interface", "Input dimensions couldn't be viewed as 2D.");
    return input.reshape(shape);
}
Tensor to3(Tensor input)
{
    auto shape = input.shape();
    if (shape.size() == 2) shape.insert(shape.begin(), 1);
    SpykerCompare(shape.size(), ==, 3, "Interface", "Input dimensions couldn't be viewed as 3D.");
    return input.reshape(shape);
}
Tensor to4(Tensor input)
{
    auto shape = input.shape();
    if (shape.size() == 3) shape.insert(shape.begin(), 1);
    SpykerCompare(shape.size(), ==, 4, "Interface", "Input dimensions couldn't be viewed as 4D.");
    return input.reshape(shape);
}
Tensor to5(Tensor input)
{
    auto shape = input.shape();
    if (shape.size() == 4) shape.insert(shape.begin(), 1);
    SpykerCompare(shape.size(), ==, 5, "Interface", "Input dimensions couldn't be viewed as 5D.");
    return input.reshape(shape);
}

Expand2::Expand2(const Shape &shape) : data(shape)
{
    if (shape.size() == 1)
        data = {data[0], data[0]};
    else
    {
        SpykerCompare(shape.size(), ==, 2, "Interface::Expand2", "Incorrect input shape.");
    }
}

Expand4::Expand4(const Shape &shape) : data(shape)
{
    if (shape.size() == 1)
        data = {data[0], data[0], data[0], data[0]};
    else if (shape.size() == 2)
        data = {data[0], data[1], data[0], data[1]};
    else
    {
        SpykerCompare(shape.size(), ==, 4, "Interface::Expand2", "Incorrect input shape.");
    }
}

Tensor canny(Tensor input, Scalar low, Scalar high)
{
    input = to4(input);
    Tensor output(input.device(), Type::U8, input.shape());
    return canny(input, output, low, high);
}

Tensor canny(Tensor input, Tensor output, Scalar low, Scalar high)
{
    input = to4(input), output = to4(output);

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Canny", "Tensor devices are not compatible.");
    SpykerCompare(input.type(), ==, output.type(), "Interface::Canny", "Input and output must have the same type.");
    SpykerCompare(input.shape(), ==, output.shape(), "Interface::Canny", "Input and output must have the same shape.");

    input = input.reshape({-1, input.shape(2), input.shape(3)});
    auto out = output.reshape({-1, output.shape(2), output.shape(3)});
    Core::canny(input, out, low, high, input.device());
    return output;
}

Tensor dog(Tensor input, Tensor kernel, Tensor output, Expand4 pad)
{
    input = to4(input), output = to4(output);

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::DoG", "Tensor devices are not compatible.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::DoG", "Kernel data type must be F32.");
    SpykerCompare(output.type(), ==, Type::F32, "Interface::DoG", "Output data type must be F32.");
    SpykerCompare(output.shape(), ==, dogShape(input.shape(), kernel.shape(), pad), "Interface::DoG",
                  "Tensor shapes are not compatible.");

    auto out = output.reshape({-1, kernel.shape(1), output.shape(2), output.shape(3)});
    input = input.reshape({-1, 1, input.shape(2), input.shape(3)});
    kernel = kernel.reshape({-1, 1, kernel.shape(2), kernel.shape(3)});
    Core::dog(input, kernel, out, pad.get(), input.device());
    return output;
}

Tensor dog(Tensor input, Tensor kernel, Expand4 pad)
{
    input = to4(input);
    Tensor output(input.device(), Type::F32, dogShape(input.shape(), kernel.shape(), pad));
    return dog(input, kernel, output, pad);
}

Tensor gabor(Tensor input, Tensor kernel, Tensor output, Expand4 pad)
{
    input = to4(input), output = to4(output);

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::Gabor", "Tensor devices are not compatible.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::Gabor", "Kernel data type must be F32.");
    SpykerCompare(output.type(), ==, Type::F32, "Interface::Gabor", "Output data type must be F32.");
    SpykerCompare(output.shape(), ==, gaborShape(input.shape(), kernel.shape(), pad), "Interface::Gabor",
                  "Tensor shapes are not compatible.");

    auto out = output.reshape({-1, kernel.shape(0), output.shape(2), output.shape(3)});
    input = input.reshape({-1, 1, input.shape(2), input.shape(3)});
    kernel = kernel.reshape({-1, 1, kernel.shape(1), kernel.shape(2)});
    Core::gabor(input, kernel, out, pad.get(), input.device());
    return output;
}

Tensor gabor(Tensor input, Tensor kernel, Expand4 pad)
{
    input = to4(input);
    Tensor output(input.device(), Type::F32, gaborShape(input.shape(), kernel.shape(), pad));
    return gabor(input, kernel, output, pad);
}

Tensor log(Tensor input, Tensor kernel, Tensor output, Expand4 pad)
{
    input = to4(input), output = to4(output);

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::LoG", "Tensor devices are not compatible.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::LoG", "Kernel data type must be F32.");
    SpykerCompare(output.type(), ==, Type::F32, "Interface::LoG", "Output data type must be F32.");
    SpykerCompare(output.shape(), ==, logShape(input.shape(), kernel.shape(), pad), "Interface::LoG",
                  "Tensor shapes are not compatible.");

    input = input.reshape({-1, 1, input.shape(2), input.shape(3)});
    kernel = kernel.reshape({-1, 1, kernel.shape(2), kernel.shape(3)});
    auto out = output.reshape({-1, kernel.shape(0), output.shape(2), output.shape(3)});
    Core::log(input, kernel, out, pad.get(), input.device());
    return output;
}

Tensor log(Tensor input, Tensor kernel, Expand4 pad)
{
    input = to4(input);
    Tensor output(input.device(), Type::F32, logShape(input.shape(), kernel.shape(), pad));
    return log(input, kernel, output, pad);
}

void zcaFit(Tensor input, Tensor mean, Tensor trans, Scalar epsilon, bool transform)
{
    input = least2(input);

    bool compat = Core::compatible({input.device(), mean.device(), trans.device()});
    SpykerAssert(compat, "Interface::ZCA", "Tensor devices are not compatible.");
    SpykerCompare(input.type(), ==, Type::F32, "Interface::ZCA", "Input data type must be F32.");
    SpykerCompare(mean.type(), ==, Type::F32, "Interface::ZCA", "Mean data type must be F32.");
    SpykerCompare(trans.type(), ==, Type::F32, "Interface::ZCA", "Transform data type must be F32.");

    input = input.reshape({input.shape(0), -1});
    zcaCheck(input.shape(), mean.shape(), trans.shape());
    Core::zca_fit(input, mean, trans, epsilon, transform, input.device());
}

Tensor zcaTrans(Tensor input, Tensor mean, Tensor trans, bool inplace)
{
    if (!inplace) input = input.copy();
    input = least2(input);

    bool compat = Core::compatible({input.device(), mean.device(), trans.device()});
    SpykerAssert(compat, "Interface::ZCA", "Tensor devices are not compatible.");
    SpykerCompare(input.type(), ==, Type::F32, "Interface::ZCA", "Input data type must be F32.");
    SpykerCompare(mean.type(), ==, Type::F32, "Interface::ZCA", "Mean data type must be F32.");
    SpykerCompare(trans.type(), ==, Type::F32, "Interface::ZCA", "Transform data type must be F32.");

    auto in = input.reshape({input.shape(0), -1});
    zcaCheck(in.shape(), mean.shape(), trans.shape());
    Core::zca_trans(in, mean, trans, in.device());
    return input;
}

Tensor zcaSplit(Tensor input, Tensor output)
{
    input = least2(input), output = least3(output);

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::ZCA", "Tensor devices are not compatible.");
    SpykerCompare(input.type(), ==, output.type(), "Interface::ZCA", "Input and output type must have the same type.");
    SpykerCompare(output.shape(), ==, zcaSplitShape(input.shape()), "Interface::ZCA", "Output shape is incorrect.");

    input = input.reshape({input.shape(0), -1});
    auto out = output.reshape({output.shape(0), 2, -1});
    Core::zca_split(input, out, input.device());
    return output;
}

Tensor zcaSplit(Tensor input)
{
    input = least2(input);
    Tensor output(input.device(), input.type(), zcaSplitShape(input.shape()));
    return zcaSplit(input, output);
}

void normalKernel(Tensor kernel, F32 mean, F32 std)
{
    kernel = kernel.reshape({-1});
    SpykerCompare(kernel.device(), ==, Kind::CPU, "Interface::Conv", "Kernel must be on CPU.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::Conv", "Kernel data type must be F32.");
    Core::normal_kernel(Dyn1(kernel), mean, std);
}

void gaussianKernel(Tensor kernel, F32 std)
{
    SpykerCompare(kernel.device(), ==, Kind::CPU, "Interface::DoG", "Kernel must be on CPU.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::DoG", "Kernel data type must be F32.");
    Core::gaussian_kernel(Dyn2(kernel), std);
}

void gaborKernel(Tensor kernel, F32 sigma, F32 theta, F32 gamma, F32 lambda, F32 psi)
{
    SpykerCompare(kernel.device(), ==, Kind::CPU, "Interface::Gabor", "Kernel must be on CPU.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::Gabor", "Kernel data type must be F32.");
    Core::gabor_kernel(Dyn2(kernel), sigma, theta, gamma, lambda, psi);
}

void logKernel(Tensor kernel, F32 std)
{
    SpykerCompare(kernel.device(), ==, Kind::CPU, "Interface::LoG", "Kernel must be on CPU.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::LoG", "Kernel data type must be F32.");
    Core::log_kernel(Dyn2(kernel), std);
}

Tensor conv(Tensor input, Tensor kernel, Expand2 stride, Expand4 pad)
{
    input = to5(input);
    check("Stride", stride, {1, 1}), check("Pad", pad, {0, 0, 0, 0});
    Tensor output(input.device(), Type::F32, convShape(input.shape(), kernel.shape(), stride, pad));
    return conv(input, kernel, output, stride, pad);
}

Tensor conv(Tensor input, Tensor kernel, Tensor output, Expand2 stride, Expand4 pad)
{
    input = to5(input), output = to5(output);
    check("Stride", stride, {1, 1}), check("Pad", pad, {0, 0, 0, 0});

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::Cond", "Tensor devices are not compatible.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::Conv", "Kernel data type must be F32.");
    SpykerCompare(output.type(), ==, Type::F32, "Interface::Conv", "Output data type must be F32.");
    SpykerCompare(output.shape(), ==, convShape(input.shape(), kernel.shape(), stride, pad), "Interface::Conv",
                  "Tensor shapes are not compatible.");

    input = input.reshape({-1, input.shape(2), input.shape(3), input.shape(4)});
    auto out = output.reshape({-1, output.shape(2), output.shape(3), output.shape(4)});
    Core::conv(input, kernel, out, stride.get(), pad.get(), input.device());
    return output;
}

Tensor fc(Tensor input, Tensor kernel, bool sign)
{
    input = to3(input);
    Tensor output(input.device(), Type::F32, fcShape(input.shape(), kernel.shape()));
    return fc(input, kernel, output, sign);
}

Tensor fc(Tensor input, Tensor kernel, Tensor output, bool sign)
{
    input = to3(input), output = to3(output);

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::FC", "Tensor devices are not compatible.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::FC", "Kernel data type must be F32.");
    SpykerCompare(output.type(), ==, Type::F32, "Interface::FC", "Output data type must be F32.");
    SpykerCompare(output.shape(), ==, fcShape(input.shape(), kernel.shape()), "Interface::FC",
                  "Tensor shapes are not compatible.");

    if (!sign) Core::fc(input, kernel, output, input.device());
    if (sign) Core::signfc(input, kernel, output, input.device());
    return output;
}

Tensor pad(Tensor input, Expand4 pad, Scalar value)
{
    input = to5(input);
    check("Pad", pad, {0, 0, 0, 0});
    Tensor output(input.device(), input.type(), padShape(input.shape(), pad));
    return Spyker::pad(input, output, pad, value);
}

Tensor pad(Tensor input, Tensor output, Expand4 pad, Scalar value)
{
    input = to5(input), output = to5(output);
    check("Pad", pad, {0, 0, 0, 0});

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Pad", "Tensor devices are not compatible.");
    SpykerCompare(input.type(), ==, output.type(), "Interface::Pad", "Input and output must have the same data type.");
    SpykerCompare(output.shape(), ==, padShape(input.shape(), pad), "Interface::Pad",
                  "Tensor shapes are not compatible.");

    input = input.reshape({-1, input.shape(3), input.shape(4)});
    auto out = output.reshape({-1, output.shape(3), output.shape(4)});
    Core::pad(input, out, pad.get(), value, input.device());
    return output;
}

Tensor threshold(Tensor input, Scalar threshold, Scalar value, bool inplace)
{
    if (!inplace) input = input.copy();
    auto in = input.reshape({-1});
    Core::threshold(in, threshold, value, in.device());
    return input;
}

Tensor quantize(Tensor input, Scalar lower, Scalar middle, Scalar upper, bool inplace)
{
    if (!inplace) input = input.copy();
    auto in = input.reshape({-1});
    Core::quantize(in, lower, middle, upper, in.device());
    return input;
}

Tensor code(Tensor input, Size time, bool sort, Type type, Code code)
{
    input = least2(input);
    SpykerCompare(time, >, 0, "Interface::RankCode", "Input time can't be less than 1.");
    Tensor output(input.device(), type, codeShape(input.shape(), time));
    return Spyker::code(input, output, time, sort, code);
}

Tensor code(Tensor input, Tensor output, Size time, bool sort, Code code)
{
    input = least2(input), output = least3(output);

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::RankCode", "Tensor devices are not compatible.");
    SpykerCompare(time, >, 0, "Interface::RankCode", "Input time can't be less than 1.");
    SpykerCompare(output.shape(), ==, codeShape(input.shape(), time), "Interface::RankCode",
                  "Tensor shapes are not compatible.");

    input = input.reshape({input.shape(0), -1});
    auto out = output.reshape({output.shape(0), output.shape(1), -1});
    if (code == Code::Rank) Core::rank_code(input, out, sort, input.device());
    if (code == Code::Rate) Core::rate_code(input, out, sort, input.device());
    return output;
}

Tensor infinite(Tensor input, Scalar value, bool inplace)
{
    if (!inplace) input = input.copy();
    input = least3(input);
    auto in = input.reshape({input.shape(0), input.shape(1), -1});
    Core::rank_infinite(in, value, in.device());
    return input;
}

Tensor fire(Tensor input, Scalar threshold, Type type, Code code)
{
    Tensor output(input.device(), type, input.shape());
    return fire(input, output, threshold, code);
}

Tensor fire(Tensor input, Tensor output, Scalar threshold, Code code)
{
    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Fire", "Tensor devices are not compatible.");
    SpykerCompare(output.shape(), ==, input.shape(), "Interface::Fire", "Tensor shapes are not compatible.");

    if (code == Code::Rank)
    {
        input = input.reshape({-1});
        auto out = output.reshape({-1});
        Core::rank_fire(input, out, threshold, input.device());
    }
    if (code == Code::Rate)
    {
        if (input.dims() == 2)
        {
            input = input.reshape({1, input.shape(1), input.shape(2)});
            output = output.reshape({1, output.shape(1), output.shape(2)});
        }
        input = input.reshape({input.shape(0), input.shape(1), -1});
        auto out = output.reshape({output.shape(0), output.shape(1), -1});
        Core::rate_fire(input, out, threshold, input.device());
    }
    return output;
}

Tensor gather(Tensor input, Scalar threshold, Type type, Code code)
{
    input = least3(input);
    Tensor output(input.device(), type, gatherShape(input.shape()));
    return gather(input, output, threshold, code);
}

Tensor gather(Tensor input, Tensor output, Scalar threshold, Code code)
{
    input = least3(input), output = least2(output);

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Gather", "Tensor devices are not compatible.");
    SpykerCompare(output.shape(), ==, gatherShape(input.shape()), "Interface::Gather",
                  "Tensor shapes are not compatible.");

    input = input.reshape({input.shape(0), input.shape(1), -1});
    auto out = output.reshape({output.shape(0), -1});
    if (code == Code::Rank) Core::rank_gather(input, out, threshold, input.device());
    if (code == Code::Rate) Core::rate_gather(input, out, threshold, input.device());
    return output;
}

Tensor scatter(Tensor input, Size time, Type type)
{
    input = least2(input);
    SpykerCompare(time, >, 0, "Interface::Scatter", "Input time can't be less than 1.");
    Tensor output(input.device(), type, scatterShape(input.shape(), time));
    return scatter(input, output);
}

Tensor scatter(Tensor input, Tensor output)
{
    input = least2(input), output = least3(output);

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Scatter", "Tensor devices are not compatible.");
    SpykerCompare(output.shape(), ==, scatterShape(input.shape(), output.shape(1)), "Interface::Scatter",
                  "Tensor shapes are not compatible.");

    input = input.reshape({input.shape(0), -1});
    auto out = output.reshape({output.shape(0), output.shape(1), -1});
    Core::rank_scatter(input, out, input.device());
    return output;
}

Tensor pool(Tensor input, Expand2 kernel, Expand2 stride, Expand4 pad, Tensor rates)
{
    input = to5(input);
    if (stride.get() == Shape{0, 0}) stride = kernel;
    check("Kernel", kernel, {1, 1}), check("Stride", stride, {1, 1}), check("Pad", pad, {0, 0, 0, 0});
    Tensor output(input.device(), input.type(), poolShape(input.shape(), kernel, stride, pad));
    return pool(input, output, kernel, stride, pad, rates);
}

Tensor pool(Tensor input, Tensor output, Expand2 kernel, Expand2 stride, Expand4 pad, Tensor rates)
{
    input = to5(input), output = to5(output);
    if (stride.get() == Shape{0, 0}) stride = kernel;
    check("Kernel", kernel, {1, 1}), check("Stride", stride, {1, 1}), check("Pad", pad, {0, 0, 0, 0});

    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Pool", "Tensor devices are not compatible.");
    SpykerCompare(input.type(), ==, output.type(), "Interface::Pool", "Input and output must have the same data type.");
    SpykerCompare(output.shape(), ==, poolShape(input.shape(), kernel, stride, pad), "Interface::Pool",
                  "Tensor shapes are not compatible.");

    if (!rates)
    {
        input = input.reshape({-1, input.shape(2), input.shape(3), input.shape(4)});
        auto out = output.reshape({-1, output.shape(2), output.shape(3), output.shape(4)});
        Core::rank_pool(input, out, kernel.get(), stride.get(), pad.get(), input.device());
    }
    if (rates)
    {
        SpykerCompare(input.type(), ==, rates.type(), "Interface::Pool",
                      "Input and rates must have the same data type.");
        SpykerCompare(rates.shape(), ==, gatherShape(input.shape()), "Interface::Pool",
                      "Tensor shapes are not compatible.");
        Core::rate_pool(input, rates, output, kernel.get(), stride.get(), pad.get(), input.device());
    }

    return output;
}

Tensor inhibit(Tensor input, Scalar threshold, bool inplace)
{
    if (!inplace) input = input.copy();
    input = to5(input);
    auto in = input.reshape({input.shape(0), input.shape(1), input.shape(2), -1});
    Core::rank_inhibit(in, threshold, in.device());
    return input;
}

Winners fcwta(Tensor input, Size radius, Size count, Scalar threshold)
{
    input = to3(input);
    SpykerCompare(radius, >=, 0, "Interface::FCWTA", "Radius can't be less than 0.");
    SpykerCompare(count, >, 0, "Interface::FCWTA", "Winners count can't be less than 1.");
    return Core::rank_fcwta(input, radius, count, threshold, input.device());
}

Winners convwta(Tensor input, Expand2 radius, Size count, Scalar threshold)
{
    input = to5(input);
    check("Radius", radius, {0, 0});
    SpykerCompare(count, >, 0, "Interface::ConvWTA", "Winners count can't be less than 1.");
    return Core::rank_convwta(input, radius.get(), count, threshold, input.device());
}

void rank_fcstdp(Tensor input, Tensor kernel, Tensor output, std::vector<STDPConfig> &config, const Winners &winners)
{
    input = to3(input), output = to3(output);

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::STDP", "Tensor devices are not compatible.");
    SpykerCompare(config.size(), >, 0, "Interface::STDP", "At least one STDP Config must be given.");
    SpykerCompare(winners.size(), >, 0, "Interface::STDP", "Winners can't be empty.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::STDP", "Kernel data type must be F32.");
    SpykerCompare(input.type(), ==, output.type(), "Interface::STDP", "Input and output must have the same data type.");
    SpykerCompare(output.shape(), ==, fcShape(input.shape(), kernel.shape()), "Interface::STDP",
                  "Tensor shapes are not compatible.");

    Core::rank_fcstdp(input, kernel, output, config, winners, input.device());
}

void rank_convstdp(Tensor input, Tensor kernel, Tensor output, const std::vector<STDPConfig> &config,
                   const Winners &winners, Expand2 stride, Expand4 pad)
{
    input = to5(input), output = to5(output);

    bool compat = Core::compatible({input.device(), kernel.device(), output.device()});
    SpykerAssert(compat, "Interface::STDP", "Tensor devices are not compatible.");
    SpykerCompare(config.size(), >, 0, "Interface::STDP", "At least one STDP Config must be given.");
    SpykerCompare(winners.size(), >, 0, "Interface::STDP", "Winners can't be empty.");
    SpykerCompare(kernel.type(), ==, Type::F32, "Interface::STDP", "Kernel data type must be F32.");
    SpykerCompare(input.type(), ==, output.type(), "Interface::STDP", "Input and output must have the same data type.");
    SpykerCompare(output.shape(), ==, convShape(input.shape(), kernel.shape(), stride, pad), "Interface::STDP",
                  "Tensor shapes are not compatible.");

    Core::rank_convstdp(input, kernel, output, config, winners, stride.get(), pad.get(), input.device());
}

INIT(DoG) INIT(Gabor) INIT(LoG) INIT(Conv) INIT(FC);

DoG::DoG(Size size, const std::vector<DoGFilter> &filters, Expand4 pad) : DoG(Kind::CPU, size, filters, pad) {}

DoG::DoG(Device device, Size size, const std::vector<DoGFilter> &filters, Expand4 pad)
    : _init(true), _device(device), _pad(pad)
{
    SpykerCompare(size, >, 0, "Interface::DoG", "Input size can't be less than 1.");
    SpykerCompare(filters.size(), >, 0, "Interface::DoG", "Input filters count can't be less then 1.");
    check("Pad", _pad, {0, 0, 0, 0});

    auto count = Size(filters.size());
    kernel = Tensor(Type::F32, {2, count, 2 * size + 1, 2 * size + 1});
    for (Size i = 0; i < count; ++i)
    {
        gaussianKernel(kernel[0][i], filters[i].std1);
        gaussianKernel(kernel[1][i], filters[i].std2);
    }
    kernel = kernel.to(_device);
}

Tensor DoG::operator()(Tensor input, Tensor output)
{
    SpykerCompare(_init, ==, true, "Interface::DoG", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::DoG", "Input device is not the same as layer's.");
    dog(input, kernel, output, _pad);
    return output;
}

Tensor DoG::operator()(Tensor input)
{
    SpykerCompare(_init, ==, true, "Interface::DoG", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::DoG", "Input device is not the same as layer's.");
    return dog(input, kernel, _pad);
}

Gabor::Gabor(Size size, const std::vector<GaborFilter> &filters, Expand4 pad) : Gabor(Kind::CPU, size, filters, pad) {}

Gabor::Gabor(Device device, Size size, const std::vector<GaborFilter> &filters, Expand4 pad)
    : _init(true), _device(device), _pad(pad)
{
    SpykerCompare(size, >, 0, "Interface::Gabor", "Input size can't be less than 1.");
    SpykerCompare(filters.size(), >, 0, "Interface::Gabor", "Input filters count can't be less then 1.");
    check("Pad", _pad, {0, 0, 0, 0});

    auto count = Size(filters.size());
    kernel = Tensor(Type::F32, {count, 2 * size + 1, 2 * size + 1});
    for (Size i = 0; i < count; ++i)
    {
        auto filter = filters[i];
        gaborKernel(kernel[i], filter.sigma, filter.theta, filter.gamma, filter.lambda, filter.psi);
    }
    kernel = kernel.to(_device);
}

Tensor Gabor::operator()(Tensor input, Tensor output)
{
    SpykerCompare(_init, ==, true, "Interface::Gabor", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::Gabor", "Input device is not the same as layer's.");
    gabor(input, kernel, output, _pad);
    return output;
}

Tensor Gabor::operator()(Tensor input)
{
    SpykerCompare(_init, ==, true, "Interface::Gabor", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::Gabor", "Input device is not the same as layer's.");
    return gabor(input, kernel, _pad);
}

LoG::LoG(Size size, const std::vector<F32> &stds, Expand4 pad) : LoG(Kind::CPU, size, stds, pad) {}

LoG::LoG(Device device, Size size, const std::vector<F32> &stds, Expand4 pad) : _init(true), _device(device), _pad(pad)
{
    SpykerCompare(size, >, 0, "Interface::LoG", "Input size can't be less than 1.");
    SpykerCompare(stds.size(), >, 0, "Interface::LoG", "Input stds count can't be less then 1.");
    check("Pad", _pad, {0, 0, 0, 0});

    auto count = Size(stds.size());
    kernel = Tensor(Type::F32, {2, count, 2 * size + 1, 2 * size + 1});
    for (Size i = 0; i < count; ++i)
    {
        gaussianKernel(kernel[0][i], stds[i] / std::sqrt(2));
        gaussianKernel(kernel[1][i], stds[i] * std::sqrt(2));
    }
    kernel = kernel.to(_device);
}

Tensor LoG::operator()(Tensor input, Tensor output)
{
    SpykerCompare(_init, ==, true, "Interface::LoG", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::LoG", "Input device is not the same as layer's.");
    log(input, kernel, output, _pad);
    return output;
}

Tensor LoG::operator()(Tensor input)
{
    SpykerCompare(_init, ==, true, "Interface::LoG", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::LoG", "Input device is not the same as layer's.");
    return log(input, kernel, _pad);
}

ZCA::ZCA() {}

ZCA &ZCA::fit(Tensor input, Scalar epsilon, bool transform)
{
    SpykerCompare(input.dims(), >=, 2, "Interface::ZCA", "Input dimensions must be at least 2D.");
    input = input.reshape({input.shape(0), -1});
    mean = Tensor(Type::F32, {input.shape(1)});
    this->transform = Tensor(Type::F32, {input.shape(1), input.shape(1)});
    zcaFit(input, mean, this->transform, epsilon, transform);
    return *this;
}

Tensor ZCA::operator()(Tensor input, bool inplace)
{
    SpykerCompare(bool(mean), ==, true, "Interface::ZCA", "Layer is not initialized.");
    SpykerCompare(bool(transform), ==, true, "Interface::ZCA", "Layer is not initialized.");
    return zcaTrans(input, mean, transform, inplace);
}

Tensor ZCA::split(Tensor input, Tensor output)
{
    input = least2(input), output = least3(output);
    zcaSplit(input, output);
    return output;
}

Tensor ZCA::split(Tensor input) { return zcaSplit(input); }

Conv::Conv(Size input, Size output, Expand2 kernel, Expand2 stride, Expand4 pad, F32 mean, F32 std)
    : Conv(Kind::CPU, input, output, kernel, stride, pad, mean, std)
{
}

Conv::Conv(Device device, Size input, Size output, Expand2 _kernel, Expand2 stride, Expand4 pad, F32 mean, F32 std)
    : _init(true), _device(device), _stride(stride), _pad(pad)
{
    SpykerCompare(input, >, 0, "Interface::Conv", "Input size can't be less than 1.");
    SpykerCompare(output, >, 0, "Interface::Conv", "Output size can't be less than 1.");
    check("Kernel", _kernel, {1, 1}), check("Stride", _stride, {1, 1}), check("Pad", _pad, {0, 0, 0, 0});

    kernel = Tensor(Type::F32, {output, input, _kernel[0], _kernel[1]});
    normalKernel(kernel, mean, std);
    kernel = kernel.to(_device);
}

Tensor Conv::operator()(Tensor input, Tensor output)
{
    SpykerCompare(_init, ==, true, "Interface::Conv", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::Conv", "Input device is not the same as layer's.");
    conv(input, kernel, output, _stride.get(), _pad.get());
    return output;
}

Tensor Conv::operator()(Tensor input)
{
    SpykerCompare(_init, ==, true, "Interface::Conv", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::Conv", "Input device is not the same as layer's.");
    return conv(input, kernel, _stride.get(), _pad.get());
}

SparseTensor Conv::operator()(SparseTensor input, Scalar threshold)
{
    SpykerCompare(_init, ==, true, "Interface::Conv", "Layer is used before initializing.");
    SpykerCompare(_device, ==, Kind::CPU, "Interface::Conv", "Input device is not the same as layer's.");
    return Sparse::conv(input, kernel, threshold, _stride, _pad);
}

void Conv::stdp(Tensor input, const Winners &winners, Tensor output)
{
    SpykerCompare(_init, ==, true, "Interface::STDP", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::STDP", "Input device is not the same as layer's.");
    SpykerCompare(stdpconfig.size(), >, 0, "Interface::STDP", "At least one STDP Config must be given.");
    SpykerCompare(winners.size(), >, 0, "Interface::STDP", "Winners can't be empty.");
    rank_convstdp(input, kernel, output, stdpconfig, winners, _stride.get(), _pad.get());
}

void Conv::stdp(SparseTensor input, const Winners &winners)
{
    SpykerCompare(stdpconfig.size(), >, 0, "Interface::STDP", "At least one STDP Config must be given.");
    SpykerCompare(winners.size(), >, 0, "Interface::STDP", "Winners can't be empty.");
    SpykerCompare(_init, ==, true, "Interface::STDP", "Layer is used before initializing.");
    SpykerCompare(_device, ==, Kind::CPU, "Interface::Conv", "Input device is not the same as layer's.");
    Core::sparse_stdp(*(Sparse5 *)input.data(), kernel, stdpconfig, winners, _pad.get());
}

FC::FC(Size input, Size output, F32 mean, F32 std) : FC(Kind::CPU, input, output, mean, std) {}

FC::FC(Device device, Size input, Size output, F32 mean, F32 std) : _init(true), _device(device)
{
    SpykerCompare(input, >, 0, "Interface::FC", "Input size can't be less than 1.");
    SpykerCompare(output, >, 0, "Interface::FC", "Output size can't be less than 1.");

    kernel = Tensor(Type::F32, {output, input});
    normalKernel(kernel, mean, std);
    kernel = kernel.to(_device);
}

Tensor FC::operator()(Tensor input, Tensor output, bool sign)
{
    SpykerCompare(_init, ==, true, "Interface::FC", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::FC", "Input device is not the same as layer's.");
    fc(input, kernel, output, sign);
    return output;
}

Tensor FC::operator()(Tensor input, bool sign)
{
    SpykerCompare(_init, ==, true, "Interface::FC", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::FC", "Input device is not the same as layer's.");
    return fc(input, kernel, sign);
}

Tensor FC::backward(Tensor input, Tensor output, Tensor grad)
{
    Tensor next(_device, grad.type(), input.shape());
    return backward(input, output, grad, next);
}

Tensor FC::backward(Tensor input, Tensor output, Tensor grad, Tensor next)
{
    input = to2(input), output = to2(output), grad = to2(grad), next = to2(next);
    SpykerCompare(_init, ==, true, "Interface::FC", "Layer is used before initializing.");
    bool compat = Core::compatible({input.device(), output.device(), grad.device(), _device});
    SpykerAssert(compat, "Interface::Backward", "Tensor devices are not compatible.");
    Core::fcbackward(kernel, input, output, grad, next, bpconfig, _device);
    return next;
}

void FC::stdp(Tensor input, const Winners &winners, Tensor output)
{
    input = to3(input), output = to3(output);
    SpykerCompare(stdpconfig.size(), >, 0, "Interface::STDP", "At least one STDP Config must be given.");
    SpykerCompare(winners.size(), >, 0, "Interface::STDP", "Winners can't be empty.");
    SpykerCompare(_init, ==, true, "Interface::STDP", "Layer is used before initializing.");
    SpykerCompare(input.device(), ==, _device, "Interface::STDP", "Input device is not the same as layer's.");
    rank_fcstdp(input, kernel, output, stdpconfig, winners);
}

Tensor backward(Tensor input, Tensor target, Size time, Scalar gamma)
{
    Tensor output(input.device(), input.type(), input.shape());
    return backward(input, output, target, time, gamma);
}

Tensor backward(Tensor input, Tensor output, Tensor target, Size time, Scalar gamma)
{
    SpykerCompare(target.type(), ==, Type::I64, "Interface::Backward", "Target type must have I64 type.");
    SpykerCompare(time, >, 0, "Interface::Backward", "Time steps must be higher than zero.");
    SpykerCompare(input.shape(), ==, output.shape(), "Interface::Backward",
                  "Output must have the same shape as input.");
    SpykerCompare(input.shape(0), ==, target.shape(0), "Interface::Backward",
                  "Input and target shapes are not compatible.");

    bool compat = Core::compatible({input.device(), output.device(), target.device()});
    SpykerAssert(compat, "Interface::Backward", "Tensor devices are not compatible.");
    Core::backward(input, output, target, time, gamma, input.device());
    return output;
}

Tensor labelize(Tensor input, Scalar threshold)
{
    Tensor output(input.device(), Type::I64, {input.shape(0)});
    return labelize(input, output, threshold);
}

Tensor labelize(Tensor input, Tensor output, Scalar threshold)
{
    SpykerCompare(output.type(), ==, Type::I64, "Interface::Labelize", "Output tensor must have I64 type.");
    bool compat = Core::compatible({input.device(), output.device()});
    SpykerAssert(compat, "Interface::Labelize", "Tensor devices are not compatible.");
    Core::labelize(input, output, threshold, input.device());
    return output;
}

#define GET(tensor) *(Sparse5 *)tensor.data()

namespace Sparse
{
SparseTensor conv(SparseTensor input, Tensor kernel, Scalar threshold, Expand2 stride, Expand4 pad)
{
    check("Stride", stride, {1, 1}), check("Pad", pad, {0, 0, 0, 0});
    SparseTensor output(convShape(input.shape(), kernel.shape(), {1, 1}, pad));
    Core::sparse_conv(GET(input), kernel, threshold, GET(output), stride.get(), pad.get());
    return output;
}

SparseTensor pad(SparseTensor input, Expand4 pad)
{
    check("Pad", pad, {0, 0, 0, 0});
    SparseTensor output(padShape(input.shape(), pad));
    Core::sparse_pad(GET(input), GET(output), pad.get());
    return output;
}

SparseTensor code(Tensor input, Size time, bool sort)
{
    input = to4(input);
    SpykerCompare(time, >, 0, "Interface::RateCvt", "Input time can't be less than 1.");
    SparseTensor output({input.shape(0), time, input.shape(1), input.shape(2), input.shape(3)});
    Core::sparse_code(input, GET(output), sort);
    return output;
}

Tensor gather(SparseTensor input, Type type)
{
    auto shape = input.shape();
    Tensor output(type, {shape[0], shape[2], shape[3], shape[4]});
    return gather(input, output);
}

Tensor gather(SparseTensor input, Tensor output)
{
    Core::sparse_gather(GET(input), output);
    return output;
}

SparseTensor pool(SparseTensor input, Expand2 kernel, Expand2 stride, Expand4 pad)
{
    if (stride.get() == Shape{0, 0}) stride = kernel;
    check("Kernel", kernel, {1, 1}), check("Stride", stride, {1, 1}), check("Pad", pad, {0, 0, 0, 0});
    SparseTensor output(poolShape(input.shape(), kernel, stride, pad));
    Core::sparse_pool(GET(input), GET(output), kernel.get(), stride.get(), pad.get());
    return output;
}

SparseTensor inhibit(SparseTensor input)
{
    SparseTensor output(input.shape());
    Core::sparse_inhibit(GET(input), GET(output));
    return output;
}

Winners convwta(SparseTensor input, Expand2 radius, Size count)
{
    check("Radius", radius, {0, 0});
    SpykerCompare(count, >, 0, "Interface::ConvWTA", "Winners count can't be less than 1.");
    return Core::sparse_convwta(GET(input), radius.get(), count);
}
}  // namespace Sparse
}  // namespace Spyker
