#include <spyker/spyker.h>

#include <iostream>

using namespace std;
using namespace Spyker;

int main()
{
    auto temp = Tensor(Type::F32, {1, 1, 10, 10});
    auto input = Sparse::code(temp, 5);
    Conv layer(1, 1, 3, 1, 1);

    auto output1 = Sparse::conv(input, layer.kernel, 1, 1, 1);
    auto output2 = Sparse::conv(input, layer.kernel, 1, 1, 1);

    cout << output1.numel() << " " << output2.numel() << endl;
}
