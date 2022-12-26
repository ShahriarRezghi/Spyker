#include <spyker/spyker.h>

#include <iostream>

using namespace std;
using namespace Spyker;

int main()
{
    auto input = Tensor(Kind::CUDA, Type::F16, {32, 1, 32, 20, 20});
    Conv layer(Kind::CUDA, 32, 32, 3);
    layer(input);
}
