#include "impl.h"

#include <ctime>

namespace Spyker
{
std::mt19937 Generator(time(nullptr));

Timer TTT;

namespace Core
{
void random_seed(Size seed) { Generator = std::mt19937(seed); }

#ifndef SPYKER_USE_CUDA
bool cuda_available() { return false; }

Size cuda_device_count() { return 0; }

void cuda_set_device(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_current_device() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

std::vector<Size> cuda_arch_list() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

std::vector<Size> cuda_device_arch() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_total() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_free() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_taken(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

Size cuda_memory_used(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_sync() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

bool cuda_cache_enabled(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_cache_enable(bool enabled, Size device)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void cuda_cache_clear(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_cache_print(Size device) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_conv_clear() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_poisson_clear() { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_conv_options(Size light, Size heuristic, Size force)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void *pinned_alloc(Size size) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void pinned_dealloc(void *data) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void *unified_alloc(Size size) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void unified_dealloc(void *data) { SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build."); }

void cuda_copy(Size size, void *input, void *output)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void cpu2cuda(Size size, void *input, void *output)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}

void cuda2cpu(Size size, void *input, void *output)
{
    SpykerAssert(false, "Core::CUDA", "CUDA is not enabled in this build.");
}
#endif

CudaSwitch::CudaSwitch(Device device)
{
    enable = (device == Kind::CUDA);
    if (!enable) return;
    previous = cuda_current_device();
    cuda_set_device(device.index());
}

CudaSwitch::~CudaSwitch()
{
    if (enable) cuda_set_device(previous);
}

void transfer(Size size, void *input, Device source, void *output, Device destin)
{
    CudaSwitch sw(source != Kind::CUDA ? destin : source);
    if (source == Kind::CPU && destin == Kind::CUDA)
        cpu2cuda(size, input, output);
    else if (source == Kind::CUDA && destin == Kind::CPU)
        cuda2cpu(size, input, output);
    else if (source == Kind::CUDA && destin == Kind::CUDA)
        cuda_copy(size, input, output);
}

bool compatible(const std::vector<Device> &devices)
{
    if (devices.empty()) return false;
    auto kind = devices.front().kind();
    auto index = devices.front().index();

    for (auto device : devices)
    {
        if (device.kind() != kind) return false;
        if (device.index() != index && device != Kind::CUDA) return false;
    }
    return true;
}

std::vector<Device> devices()
{
    std::vector<Device> devices = {Device(Kind::CPU)};
    Size count = cuda_device_count();
    for (Size i = 0; i < count; ++i) devices.push_back(Device(Kind::CUDA, i));
    return devices;
}
}  // namespace Core
}  // namespace Spyker
