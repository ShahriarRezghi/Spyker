#include <memory>
#include <vector>

#include "base.h"

namespace Spyker
{
namespace Core
{
namespace Sparse
{
struct Point
{
    U16 k, o;
    Point(U16 k, U16 o) : k(k), o(o) {}
};

void conv_init(Vec1<U16> length, Vec2<Point> index, Len4 input, Len4 kernel, Len4 output, Len4 pad)
{
    fill(length, U16(0));

    Size si = pad.t, ei = si + input.y;
    Size sj = pad.z, ej = sj + input.x;

    for (Size i = 0; i < output.y; ++i)
        for (Size k = 0; k < kernel.y; ++k)
            for (Size t = 0; t < kernel.x; ++t)
                for (Size j = 0; j < output.x; ++j)
                {
                    Size idx = i + k, jdx = j + t;
                    bool X = bool(si <= idx) & bool(idx < ei);
                    bool Y = bool(sj <= jdx) & bool(jdx < ej);

                    if (X & Y)
                    {
                        Size row = (idx - si) * input.x + (jdx - sj);
                        *index(row, length(row)++) = Point(k * kernel.x + t, i * output.x + j);
                    }
                }
}

void conv_run(U16 length, PTR(Point, index), PTR(F32, kernel), PTR(F32, output))
{
    for (U16 i = 0; i < length; ++i) output[index[i].o] += kernel[index[i].k];
}

void conv_run(Vec1<U16> length, Vec2<Point> index, Sparse5 input, Vec3<F32> kernel, F32 threshold, Sparse5 output,
              Vec2<F32> temp, Size i, Size c)
{
    fill(temp, F32(0));

    for (Size j = 0; j < input.t; ++j)
    {
        for (Spridx idx : input(i, j))
        {
            Size row = idx.y * input.x + idx.x;
            conv_run(length(row), index(row).data, kernel(idx.z).data, temp.data);
        }

        Sparse3 &od = output(i, j);
        if (input(i, j).size > 0)
            for (Size k = 0; k < temp.y; ++k)
            {
                F32 *td = temp(k, 0);
                for (Size t = 0; t < temp.x; ++t)
                    if (td[t] > threshold)
                    {
                        od.add(Spridx(c, k, t));
                        td[t] = Limits<F32>::min();
                    }
            }
    }
}

struct Conv
{
    Vec1<U16> length;
    Vec2<Point> index;
    Len4 _input, _kernel, _output, _pad;

    Conv(Len4 input, Len4 kernel, Len4 output, Len4 pad) : _input(input), _kernel(kernel), _output(output), _pad(pad)
    {
        length = init<U16>(input.y * input.x);
        index = init<Point>(length.x, kernel.y * kernel.x);
        conv_init(length, index, input, kernel, output, pad);
    }
    ~Conv() { deinit(length, index); }

    void operator()(Sparse5 input, Vec4<F32> kernel, F32 threshold, Sparse5 output)
    {
        BatchSize(output.u);
        auto temp = init<F32>(batch_size, output.y, output.x);

#pragma omp parallel for
        for (Size i = 0; i < input.u; ++i)
            for (Size j = 0; j < kernel.t; ++j)  //
                conv_run(length, index, input, kernel(j), threshold, output, temp(BatchIndex(i)), i, j);

        deinit(temp);
    }

    bool compare(Len4 input_, Len4 kernel_, Len4 output_, Len4 pad_)
    {
        return _input.y == input_.y && _input.x == input_.x &&      //
               _kernel.y == kernel_.y && _kernel.x == kernel_.x &&  //
               _output.y == output_.y && _output.x == output_.x && _pad == pad_;
    }
};

std::vector<std::shared_ptr<Conv>> conv_handle;

void conv_clear() { conv_handle.clear(); }

Conv &conv_find(Len4 input, Len4 kernel, Len4 output, Len4 pad)
{
    for (auto conv : conv_handle)
        if (conv->compare(input, kernel, output, pad)) return *conv.get();
    conv_handle.push_back(std::shared_ptr<Conv>(new Conv(input, kernel, output, pad)));
    return *conv_handle.back().get();
}

void conv(Sparse5 input, Vec4<F32> kernel, F32 threshold, Sparse5 output, Len2 stride, Len4 pad)
{
    SpykerCompare(stride.y, ==, 1, "Core::Conv", "Stride must be 1 in sparse conv.");
    SpykerCompare(stride.x, ==, 1, "Core::Conv", "Stride must be 1 in sparse conv.");

    Len4 inlen = {input.t, input.z, input.y, input.x};
    Len4 outlen = {output.t, output.z, output.y, output.x};
    conv_find(inlen, kernel.len(), outlen, pad)(input, kernel, threshold, output);
}
}  // namespace Sparse

void sparse_conv(Sparse5 input, Dyn4 kernel, Scalar threshold, Sparse5 output, Len2 stride, Len4 pad)
{
    Sparse::conv(input, kernel, threshold, output, stride, pad);
}

void sparse_conv_clear() { Sparse::conv_handle.clear(); }
}  // namespace Core
}  // namespace Spyker
