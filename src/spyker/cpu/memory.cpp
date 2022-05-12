#include "base.h"
//
#include <cstdlib>
#include <cstring>

namespace Spyker
{
namespace Core
{
void* cpu_alloc(Size size)
{
#ifdef _WIN32
    return malloc(size);
#else
    return aligned_alloc(256, size);
#endif
}

void cpu_dealloc(void* data) { std::free(data); }

void cpu_copy(Size size, void* input, void* output) { std::memcpy(output, input, size); }

template <typename T1, typename T2>
void cpu_cast(Size size, T1* input, T2* output)
{
    std::copy(input, input + size, output);
}

void cpu_cast(Size size, Dyn input, Dyn output)
{
    IfType(T1, input.type, IfType(T2, output.type, cpu_cast(size, input.to<T1>(), output.to<T2>())));
}

template <typename T>
void cpu_fill(Size size, T* data, T value)
{
    std::fill(data, data + size, value);
}

void cpu_fill(Size size, Dyn data, Scalar value) { IfType(T, data.type, cpu_fill<T>(size, data.to<T>(), value)); }

namespace CPU
{
#ifdef SPYKER_USE_DNNL
std::unique_ptr<onednn> onednn_static;
#endif
}  // namespace CPU
}  // namespace Core
}  // namespace Spyker
