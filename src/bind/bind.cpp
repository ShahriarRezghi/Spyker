#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//
#include <spyker/base.h>
#include <spyker/config.h>
#include <spyker/helper/helper.h>
#include <spyker/opers.h>
#include <spyker/shape.h>

#include <memory>

using namespace Spyker;
namespace py = pybind11;

Kind str2kind(std::string kind)
{
    for (auto &c : kind) c = std::tolower(c);
    if (kind == "cpu") return Kind::CPU;
    if (kind == "cuda") return Kind::CUDA;
    SpykerAssert(false, "Binding", "Unknown device given.");
}

PYBIND11_MODULE(spyker_plugin, m)
{
    auto version = m.def_submodule("version");
    version.attr("major") = SPYKER_VERSION_MAJOR;
    version.attr("minor") = SPYKER_VERSION_MINOR;
    version.attr("patch") = SPYKER_VERSION_PATCH;

    m.def_submodule("control")
        .def("random_seed", &randomSeed)
        .def("cuda_available", &cudaAvailable)
        .def("cuda_device_count", &cudaDeviceCount)
        .def("cuda_set_device", &cudaSetDevice)
        .def("cuda_current_device", &cudaCurrentDevice)
        .def("cuda_arch_list", &cudaArchList)
        .def("cuda_device_arch", &cudaDeviceArch)
        .def("cuda_memory_total", &cudaMemoryTotal)
        .def("cuda_memory_free", &cudaMemoryFree)
        .def("cuda_memory_taken", &cudaMemoryTaken)
        .def("cuda_memory_used", &cudaMemoryUsed)
        .def("cuda_cache_enabled", &cudaCacheEnabled)
        .def("cuda_cache_enable", &cudaCacheEnable)
        .def("cuda_cache_clear", &cudaCacheClear)
        .def("clear_context", &clearContext)
        .def("cuda_cache_print", &cudaCachePrint)
        .def("light_conv", &lightConv)
        .def("all_devices", &allDevices)
        .def("max_threads", &maxThreads)
        .def("set_max_threads", &setMaxThreads)
        .def("set_max_threads", []() { return setMaxThreads(); })
        .def("cuda_memory_taken", []() { return cudaMemoryTaken(); })
        .def("cuda_memory_used", []() { return cudaMemoryUsed(); })
        .def("cuda_cache_enabled", []() { return cudaCacheEnabled(); })
        .def("cuda_cache_enable", [](bool enable) { return cudaCacheEnable(enable); })
        .def("cuda_cache_clear", []() { return cudaCacheClear(); })
        .def("cuda_cache_print", []() { return cudaCachePrint(); });

    py::enum_<Type>(m, "type")  //
        .value("i8", Type::I8)
        .value("i16", Type::I16)
        .value("i32", Type::I32)
        .value("i64", Type::I64)
        .value("u8", Type::U8)
        .value("u16", Type::U16)
        .value("u32", Type::U32)
        .value("u64", Type::U64)
        .value("f16", Type::F16)
        .value("f32", Type::F32)
        .value("f64", Type::F64)
        .export_values();

    py::class_<Winner>(m, "Winner")
        .def(py::init<Size, Size, Size>())
        .def_readwrite("c", &Winner::c)
        .def_readwrite("z", &Winner::z)
        .def_readwrite("t", &Winner::t)
        .def_readwrite("y", &Winner::y)
        .def_readwrite("x", &Winner::x);

    py::class_<Device>(m, "device")
        .def(py::init([](std::string kind) { return new Device(str2kind(kind)); }))
        .def(py::init([](std::string kind, I32 index) { return new Device(str2kind(kind), index); }))
        .def_property_readonly("kind",
                               [](const Device &device) {
                                   if (device.kind() == Kind::CPU) return std::string("cpu");
                                   if (device.kind() == Kind::CUDA) return std::string("cuda");
                                   return std::string("unknown");
                               })
        .def_property_readonly("index", &Device::index)
        .def("__eq__", [](const Device &device, const Device &other) { return device == other; })
        .def("__eq__", [](const Device &device, std::string kind) { return device == str2kind(kind); });

    m.def("create_tensor", [](Device device, Type type, Shape shape, bool pinned, bool unified) {
        return new Tensor(device, type, shape, pinned, unified);
    });
    m.def("create_tensor", [](I64 data, Device device, Type type, Shape shape, bool pinned, bool unified) {
        return new Tensor(std::shared_ptr<void>((void *)data, [](void *) {}), device, type, shape, pinned, unified);
    });

    py::class_<Tensor>(m, "tensor", py::buffer_protocol())
        .def(py::init())
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("bytes", &Tensor::bytes)
        .def_property_readonly("dims", &Tensor::dims)
        .def_property_readonly("dtype", &Tensor::type)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("shape", [](Tensor &tensor) { return tensor.shape(); })

        .def("pinned", &Tensor::pinned)
        .def("unified", &Tensor::unified)
        .def("__bool__", &Tensor::operator bool)
        .def("data", [](Tensor &tensor) { return (I64)tensor.data(); })
        .def("i8", &Tensor::i8)
        .def("i16", &Tensor::i16)
        .def("i32", &Tensor::i32)
        .def("i64", &Tensor::i64)
        .def("u8", &Tensor::u8)
        .def("u16", &Tensor::u16)
        .def("u32", &Tensor::u32)
        .def("u64", &Tensor::u64)
        .def("f16", &Tensor::f16)
        .def("f32", &Tensor::f32)
        .def("f64", &Tensor::f64)
        .def("cpu", &Tensor::cpu)
        .def("cuda", &Tensor::cuda)
        .def("reshape", &Tensor::reshape)
        .def("__getitem__", &Tensor::operator[])
        .def("fill", [](Tensor &tensor, F64 data) { return tensor.fill(data); })
        .def("copy", &Tensor::copy)
        .def("to", [](Tensor &tensor, Type type) { return tensor.to(type); })
        .def("to", [](Tensor &tensor, Device device) { return tensor.to(device); })
        .def("to", [](Tensor &tensor, const Tensor &other) { return tensor.to(other); })
        .def("from_", &Tensor::from)

        .def_buffer([](Tensor &tensor) {
            SpykerCompare(tensor.device(), ==, Kind::CPU, "Binding",
                          "Can't create a python buffer from non-cpu tensor.");

            Type type = tensor.type();
            Size dims = tensor.dims();
            Shape shape = tensor.shape();
            Size size = TypeSize(type);

            std::string format;
            IfType(T, type, format = py::format_descriptor<T>::format());

            Shape stride(dims, size);
            for (Size i = dims - 1; i > 0; --i) stride[i - 1] = stride[i] * shape[i];
            return py::buffer_info(tensor.data(), size, format, dims, shape, stride);
        });

    py::class_<SparseTensor>(m, "sparse_tensor")
        .def(py::init<Tensor, F64>())
        .def_property_readonly("numel", &SparseTensor::numel)
        .def_property_readonly("bytes", &SparseTensor::bytes)
        .def_property_readonly("dims", &SparseTensor::dims)
        .def_property_readonly("shape", [](SparseTensor &tensor) { return tensor.shape(); })
        .def("__bool__", &SparseTensor::operator bool)
        .def("copy", &SparseTensor::copy)
        .def("dense", &SparseTensor::dense)
        .def("sparsity", &SparseTensor::sparsity);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    m.def_submodule("shape")
        .def("zca_split", &zcaSplitShape)
        .def("dog", &dogShape)
        .def("gabor", &gaborShape)
        .def("log", &logShape)
        .def("code", &codeShape)
        .def("pad", &padShape)
        .def("gather", &gatherShape)
        .def("scatter", &scatterShape)
        .def("conv", &convShape)
        .def("pool", &poolShape)
        .def("fc", &fcShape);

    m.def_submodule("helper")  //
        .def("mnist_data", &Helper::mnistData)
        .def("mnist_label", &Helper::mnistLabel);

    py::class_<DoGFilter>(m, "DoGFilter")
        .def(py::init<F32, F32>())
        .def_readwrite("std1", &DoGFilter::std1)
        .def_readwrite("std2", &DoGFilter::std2);

    py::class_<GaborFilter>(m, "GaborFilter")
        .def(py::init<F32, F32, F32, F32, F32>())
        .def_readwrite("sigma", &GaborFilter::sigma)
        .def_readwrite("theta", &GaborFilter::theta)
        .def_readwrite("gamma", &GaborFilter::gamma)
        .def_readwrite("lambda_", &GaborFilter::lambda)
        .def_readwrite("psi", &GaborFilter::psi);

    py::class_<STDPConfig>(m, "STDPConfig")
        .def(py::init<F32, F32, bool, F32, F32>())
        .def_readwrite("stabilize", &STDPConfig::stabilize)
        .def_readwrite("pos", &STDPConfig::pos)
        .def_readwrite("neg", &STDPConfig::neg)
        .def_readwrite("low", &STDPConfig::low)
        .def_readwrite("high", &STDPConfig::high);

    py::class_<BPConfig>(m, "BPConfig")
        .def(py::init<F32, F32, F32, F32>())
        .def_readwrite("sfactor", &BPConfig::sfactor)
        .def_readwrite("lrate", &BPConfig::lrate)
        .def_readwrite("lrf", &BPConfig::lrf)
        .def_readwrite("lambda_", &BPConfig::lambda);

    py::class_<ZCA>(m, "ZCA")
        .def(py::init())
        .def(py::init<Tensor, F64, bool>())
        .def_readwrite("mean", &ZCA::mean)
        .def_readwrite("transform", &ZCA::transform)
        .def("__call__", [](ZCA &layer, Tensor input) { layer(input); })
        .def_static("split", [](Tensor input, Tensor output) { ZCA::split(input, output); })
        .def("fit",
             [](ZCA &layer, Tensor input, F64 epsilon, bool transform) { layer.fit(input, epsilon, transform); });

    py::class_<DoG>(m, "DoG")
        .def(py::init<Device, Size, const std::vector<DoGFilter> &, Shape>())
        .def_readwrite("kernel", &DoG::kernel)
        .def("__call__", [](DoG &layer, Tensor input, Tensor output) { layer(input, output); });

    py::class_<Gabor>(m, "Gabor")
        .def(py::init<Device, Size, const std::vector<GaborFilter> &, Shape>())
        .def_readwrite("kernel", &Gabor::kernel)
        .def("__call__", [](Gabor &layer, Tensor input, Tensor output) { layer(input, output); });

    py::class_<LoG>(m, "LoG")
        .def(py::init<Device, Size, const std::vector<F32> &, Shape>())
        .def_readwrite("kernel", &LoG::kernel)
        .def("__call__", [](LoG &layer, Tensor input, Tensor output) { layer(input, output); });

    py::class_<FC>(m, "FC")
        .def(py::init<Device, Size, Size, F32, F32>())
        .def_readwrite("kernel", &FC::kernel)
        .def_readwrite("config", &FC::config)
        .def_readwrite("bpconfig", &FC::bpconfig)
        .def("__call__", [](FC &layer, Tensor input, Tensor output) { return layer(input, output); })
        .def("stdp",
             [](FC &layer, Tensor input, const Winners &winners, Tensor output) { layer.stdp(input, winners, output); })
        .def("backward", [](FC &layer, Tensor input, Tensor output, Tensor grad, Tensor next) {
            layer.backward(input, output, grad, next);
        });

    py::class_<Conv>(m, "Conv")
        .def(py::init<Device, Size, Size, Shape, Shape, Shape, F32, F32>())
        .def_readwrite("kernel", &Conv::kernel)
        .def_readwrite("config", &Conv::config)
        .def("__call__", [](Conv &layer, Tensor input, Tensor output) { layer(input, output); })
        .def("stdp", [](Conv &layer, Tensor input, const Winners &winners, Tensor output) {
            layer.stdp(input, winners, output);
        });

    m.def("canny", [](Tensor input, Tensor output, F64 low, F64 high) { Spyker::canny(input, output, low, high); })
        .def("fc", [](Tensor input, Tensor kernel, Tensor output) { Spyker::fc(input, kernel, output); })
        .def("fc", [](FC &fc, Tensor input, Tensor output) { Spyker::fc(fc, input, output); })
        .def("signfc", [](Tensor input, Tensor kernel, Tensor output) { Spyker::signfc(input, kernel, output); })
        .def("signfc", [](FC &fc, Tensor input, Tensor output) { Spyker::signfc(fc, input, output); })
        .def("conv", [](Tensor input, Tensor kernel, Tensor output, Shape stride,
                        Shape pad) { Spyker::conv(input, kernel, output, stride, pad); })
        .def("conv", [](Conv &conv, Tensor input, Tensor output) { Spyker::conv(conv, input, output); })
        .def("pad", [](Tensor input, Tensor output, Shape pad, F64 value) { Spyker::pad(input, output, pad, value); })
        .def("threshold", [](Tensor input, F64 threshold, F64 value) { Spyker::threshold(input, threshold, value); })
        .def("quantize",
             [](Tensor input, F64 lower, F64 middle, F64 upper) { Spyker::quantize(input, lower, middle, upper); })
        .def("code", [](Tensor input, Tensor output, Size time, bool sort) { Spyker::code(input, output, time, sort); })
        .def("infinite", [](Tensor input, F64 value) { Spyker::infinite(input, value); })
        .def("fire", [](Tensor input, Tensor output, F64 threshold) { Spyker::fire(input, output, threshold); })
        .def("gather", [](Tensor input, Tensor output, F64 threshold) { Spyker::gather(input, output, threshold); })
        .def("scatter", [](Tensor input, Tensor output) { Spyker::scatter(input, output); })
        .def("pool", [](Tensor input, Tensor output, Shape kernel, Shape stride,
                        Shape pad) { Spyker::pool(input, output, kernel, stride, pad); })
        .def("inhibit", [](Tensor input, F64 threshold) { Spyker::inhibit(input, threshold); })
        .def("fcwta", [](Tensor input, Size radius, Size count,
                         F64 threshold) { return Spyker::fcwta(input, radius, count, threshold); })
        .def("convwta", [](Tensor input, Shape radius, Size count,
                           F64 threshold) { return Spyker::convwta(input, radius, count, threshold); })
        .def("stdp", [](Conv &conv, Tensor input, const Winners &winners,
                        Tensor output) { Spyker::stdp(conv, input, winners, output); })
        .def("stdp", [](FC &fc, Tensor input, const Winners &winners,
                        Tensor output) { Spyker::stdp(fc, input, winners, output); })
        .def("backward", [](Tensor input, Tensor output, Tensor target, Size time,
                            F64 gamma) { return Spyker::backward(input, output, target, time, gamma); })
        .def("labelize",
             [](Tensor input, Tensor output, F64 threshold) { return Spyker::labelize(input, output, threshold); });

    using namespace Spyker::Rate;
    m.def_submodule("rate")
        .def("code", [](Tensor input, Tensor output, Size time, bool sort) { Rate::code(input, output, time, sort); })
        .def("fire", [](Tensor input, Tensor output, F64 threshold) { Rate::fire(input, output, threshold); })
        .def("gather", [](Tensor input, Tensor output, F64 threshold) { Rate::gather(input, output, threshold); })
        .def("pool", [](Tensor input, Tensor rates, Tensor output, Shape kernel, Shape stride, Shape pad) {
            Rate::pool(input, rates, output, kernel, stride, pad);
        });

    using namespace Spyker::Sparse;
    m.def_submodule("sparse")
        .def("conv", [](Conv &conv, SparseTensor input, F64 threshold) { return Sparse::conv(conv, input, threshold); })
        .def("conv", [](SparseTensor input, Tensor kernel, F64 threshold, Shape stride,
                        Shape pad) { return Sparse::conv(input, kernel, threshold, stride, pad); })
        .def("pad", [](SparseTensor input, Shape pad) { return Sparse::pad(input, pad); })
        .def("code", [](Tensor input, Size time, bool sort) { return Sparse::code(input, time, sort); })
        .def("gather", [](SparseTensor input, Type type) { return Sparse::gather(input, type); })
        .def("pool", [](SparseTensor input, Shape kernel, Shape stride,
                        Shape pad) { return Sparse::pool(input, kernel, stride, pad); })
        .def("inhibit", [](SparseTensor input) { return Sparse::inhibit(input); })
        .def("convwta",
             [](SparseTensor input, Shape radius, Size count) { return Sparse::convwta(input, radius, count); })
        .def("stdp",
             [](Conv &conv, SparseTensor input, const Winners &winners) { return Sparse::stdp(conv, input, winners); });
}
