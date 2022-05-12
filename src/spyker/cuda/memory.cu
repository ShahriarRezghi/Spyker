#include "base.cuh"
//
#include <map>
#include <vector>

namespace Spyker
{
namespace Core
{
struct Cache
{
    struct Entry
    {
        void *data;
        bool free;
        Size size, rank;
    };

    bool enabled = true;
    std::vector<Entry> list;
    Size rank = 0;

    ~Cache()
    {
        for (auto &info : list) cudaFree(info.data);
    }

    std::vector<Entry>::iterator minimum()
    {
        auto rank = this->rank;
        auto entry = list.end();
        for (auto it = list.begin(); it != list.end(); ++it)
            if (it->free && it->rank < rank) rank = it->rank, entry = it;
        return entry;
    }

    void freeup(Size size)
    {
        size_t free, total;
        CudaCheck(cudaMemGetInfo(&free, &total));
        size_t margin = total * .01;
        free = (margin > free) ? 0 : free - margin;
        if (size <= free) return;

        while (size > free && !list.empty())
        {
            auto it = minimum();
            if (it == list.end()) break;
            CudaCheck(cudaFree(it->data));
            free += it->size;
            list.erase(it);
        }
    }

    void *alloc(Size size)
    {
        for (auto &entry : list)
            if (entry.free && entry.size == size)
            {
                entry.free = false;
                entry.rank = rank++;
                return entry.data;
            }

        freeup(size);
        Entry entry = {nullptr, false, size, rank++};
        CudaCheck(cudaMalloc(&entry.data, size));
        list.push_back(entry);
        return entry.data;
    }

    void dealloc(void *data)
    {
        for (auto &entry : list)
            if (entry.data == data)
            {
                entry.free = true;
                return;
            }
        CudaCheck(cudaFree(data));
    }

    void clear()
    {
        for (auto &entry : list)
            if (entry.free) CudaCheck(cudaFree(entry.data));
        auto pred = [](const Entry &entry) { return entry.free; };
        list.erase(std::remove_if(list.begin(), list.end(), pred), list.end());
    }

    Size taken()
    {
        Size total = 0;
        for (const auto &entry : list) total += entry.size;
        return total;
    }

    Size used()
    {
        Size used = 0;
        for (const auto &entry : list)
            if (!entry.free) used += entry.size;
        return used;
    }
};

std::map<Size, Cache> cuda_cache_map;

void *cuda_alloc(Size size)
{
    Cache &cache = cuda_cache_map[cuda_current_device()];
    if (cache.enabled) return cache.alloc(size);

    void *data;
    CudaCheck(cudaMalloc(&data, size));
    return data;
}

void cuda_dealloc(void *data)
{
    Cache &cache = cuda_cache_map[cuda_current_device()];
    if (cache.enabled) return cache.dealloc(data);

    cudaFree(data);
}

Size cuda_memory_total()
{
    size_t free, total;
    CudaCheck(cudaMemGetInfo(&free, &total));
    return total;
}

Size cuda_memory_free()
{
    size_t free, total;
    CudaCheck(cudaMemGetInfo(&free, &total));
    return free;
}

Size cuda_memory_taken(Size device) { return cuda_cache_map[cuda_current_device()].taken(); }

Size cuda_memory_used(Size device) { return cuda_cache_map[cuda_current_device()].used(); }

bool cuda_cache_enabled(Size device) { return cuda_cache_map[device].enabled; }

void cuda_cache_enable(bool enabled, Size device)
{
    Cache &cache = cuda_cache_map[device];
    SpykerAssert(!cache.list.empty(), "CUDA::Memory",
                 "Cache status can only be changed when there are no allocations.");
    cache.enabled = enabled;
}

void cuda_cache_clear(Size device) { cuda_cache_map[device].clear(); }

void cuda_cache_print(Size device)
{
    auto list = cuda_cache_map[device].list;
    std::sort(list.begin(), list.end(), [](const Cache::Entry &A, const Cache::Entry &B) { return A.size > B.size; });

    Size width = 0;
    for (const auto &info : list) width = std::max<Size>(width, std::to_string(info.size).size());

    for (const auto &info : list)
        std::cout << std::boolalpha << "   Cache Entry: Size = " << std::setw(width) << info.size
                  << ", Used = " << (info.free ? 'F' : 'T') << ", Pointer = " << info.data << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int device_count = -1;
int current_device = -1;

bool cuda_available()
{
    int count;
    auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) return false;
    return count != 0;
}

Size cuda_device_count()
{
    if (device_count > -1) return device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) return 0;
    return device_count;
}

void cuda_set_device(Size device)
{
    if (current_device == device) return;
    CudaCheck(cudaSetDevice(device));
}

Size cuda_current_device()
{
    if (current_device > -1) return current_device;
    CudaCheck(cudaGetDevice(&current_device));
    return current_device;
}

void *pinned_alloc(Size size)
{
    void *data;
    CudaCheck(cudaMallocHost(&data, size));
    return data;
}

void pinned_dealloc(void *data) { cudaFreeHost(data); }

void *unified_alloc(Size size)
{
    void *data;
    CudaCheck(cudaMallocManaged(&data, size));
    return data;
}

void unified_dealloc(void *data) { cudaFree(data); }

void cuda_copy(Size size, void *input, void *output)
{
    CudaCheck(cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice));
}

void cpu2cuda(Size size, void *input, void *output)
{
    CudaCheck(cudaMemcpy(output, input, size, cudaMemcpyHostToDevice));
}

void cuda2cpu(Size size, void *input, void *output)
{
    CudaCheck(cudaMemcpy(output, input, size, cudaMemcpyDeviceToHost));
}

std::vector<Size> cuda_arch_list()
{
    std::string temp;
    std::vector<Size> list;
    std::stringstream stream(SPYKER_CUDA_ARCH);
    while (std::getline(stream, temp, '-')) list.push_back(std::stoi(temp));
    return list;
}

std::vector<Size> cuda_device_arch()
{
    std::vector<Size> list;
    auto count = cuda_device_count();
    for (Size i = 0; i < count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        list.push_back(prop.major * 10 + prop.minor);
    }
    return list;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace CUDA
{
template <typename T1, typename T2>
__global__ void cast_kernel(Cize size, PTR(T1, input), PTR(T2, output))
{
    Cize idx = Index1D(T2), end = min(size, idx + Block1D(T2));
    for (Cize i = idx; i < end; i += Thread1D) cast(input[i], output[i]);
}

template <typename T1, typename T2>
void cast(Size size, T1 *input, T2 *output)
{
    cast_kernel<<<Config1D(T2, 1, 1, size)>>>(size, input, output);
}

template <typename T>
__global__ void fill_kernel(Cize size, PTR(T, input), T value)
{
    Cize idx = Index1D(T), end = min(size, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) input[i] = value;
}

template <typename T>
void fill(Size size, T *data, T value)
{
    fill_kernel<<<Config1D(T, 1, 1, size)>>>(size, data, value);
}
}  // namespace CUDA

void cuda_cast(Size size, Dyn input, Dyn output)
{
    IfType(T1, input.type, IfType(T2, output.type, CUDA::cast(size, input.to<T1>(), output.to<T2>())));
}
void cuda_fill(Size size, Dyn data, Scalar value)  //
{
    IfType(T, data.type, CUDA::fill<T>(size, data.to<T>(), value));
}
}  // namespace Core
}  // namespace Spyker
