# Table of Contents
- [Table of Contents](#table-of-contents)
- [Building From Source](#building-from-source)
  - [Prequisites](#prequisites)
    - [CUDA](#cuda)
    - [cuDNN](#cudnn)
    - [CBLAS and LAPACKE](#cblas-and-lapacke)
    - [Others](#others)
    - [Arch Linux](#arch-linux)
  - [Build and Install](#build-and-install)
    - [C++ Interface](#c-interface)
    - [Python Interface](#python-interface)
    - [Build Options](#build-options)
    - [Finding Dependencies](#finding-dependencies)

# Building From Source
In order to build Spyker from source you will need to install some prequisites, get the source code, compile the library, and install it.

## Prequisites
There are some libraries Spyker depends upon that need to be installed on your system. All of these prequisites are optional but recommended to get the best performance out of the library. Most of them can be installed using linux package managers.

### CUDA
In order to enable CUDA support in Spyker, you need to have CUDA on your system. Please refer to CUDA instalation guide [Linux]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for instructions on how to install it. CUDA version 6.5 or later is supported. 

### cuDNN
Installing cuDNN is highly recommended as it gives the performance boost needed when running networks on GPUs. Please refer to [cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) for instructions on how to install it.

### CBLAS and LAPACKE
Spyker does some linear algebra operations using CBLAS and LAPACKE libraries. Having a good implementation of these libraries can improve accuracy. Some good options are  [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) library on Intel CPUs and [OpenBLAS](https://www.openblas.net/) (usually needs to be built from source to support LAPACKE).

### Others
Spyker uses auto-vectorization to improve performance on CPUs. You will need a good and new C++ compiler to build Spyker. Some options are Clang (recommended) and GCC. Use the latest version if you can. You will also need a recent version of CMake so that it will have proper support for CUDA language.

### Arch Linux
Installing the prequisites is very easy in Arch linux using its package manager. For example:

``` shell
pacman -S base-devel clang cmake intel-mkl intel-mkl-static cuda cudnn
```

You might need sudo to run this command.

## Build and Install
First, you need to download the source code:

``` shell
git clone --recursive git@github.com:ShahriarSS/Spyker.git
cd spyker
```

Then, depending on which interface you want to build, you can begin the building process.

### C++ Interface
The C++ interface can be built like so:

``` shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
make install
```

You might need sudo to run `make install`. Check out the [Build Options](#build-options) section to customize your build.

### Python Interface
You can build the Python interface with:

``` shell
pip wheel -vvv . -w dist/
```

You will need to install the `wheel` package for Python. The built package is located in dist dictory and can be installed with `pip install <package-file-name>`. Check out the [Build Options](#build-options) section to customize your build.

### Build Options
There are some build options to give you control over the installation:

Option | Interface | Description | Type
--- | --- | --- | ---
ENABLE_TESTS | C++ | Build the library tests | CMake
ENABLE_EXAMPLES | C++ | Build the library examples | CMake
CMAKE_INSTALL_PREFIX | C++ | Set the installation directory | CMake
ENABLE_DNNL | C++, Python | Build the library with DNNL support (highly recommended) | CMake(C++), Env(Py)
ENABLE_BLAS | C++, Python | Build the library using CLBAS and LAPACKE support | CMake(C++), Env(Py)
ENABLE_CUDA | C++, Python | build the library using CUDA support | CMake(C++), Env(Py)
ENABLE_CUDNN | C++, Python | build the library using cuDNN support (highly recommended) | CMake(C++), Env(Py)
ENABLE_NATIVE | C++, Python | Build the library using native CPU instructions (must be off when building portable library) | CMake(C++), Env(Py)
ENABLE_NINJA | Python | Build the library with Ninja (pass `-G Ninja` for C++) | Env(Py)
CUDA_ARCH_LIST | C++, Python | CUDA architectures to be built for | CMake(C++), Env(Py)
MKLROOT | C++, Python | MKL root directory hint path | Env(both)
CUDA_PATH | C++, Python | CUDA root directory hint path | Env(both)
BLA_VENDOR | C++, Python | CBLAS and LAPACKE vendor ([see this](https://cmake.org/cmake/help/latest/module/FindBLAS.html)) | CMake(C++), Env(Py)

An example of setting a CMake variable:
``` shell
cmake -DVARIABLE:VALUE ...
```

An example of setting an environment variable:
``` shell
VARIABLE=VALUE pip ...
```

or by using `export` in shells. Be sure to check out the prints at the end of CMake configuration to see which options are enabled and which are disabled. 

### Finding Dependencies
finding dependencies can be tricky when it comes to a multi-platform library. Some tips are shown here that will help you find missing dependencies. So please consider these tips before creating an issue. The standard way of setting the hint paths for dependencies is listed in [Building Options](#build-options) section.

CMake package finding works by searching a list of paths for headers and library files. By default, this search is done quietly. You can enable verbose mode by setting `CMAKE_FIND_DEBUG_MODE` CMake option to `ON`. You can check if the path that contains the needed files exists and if not, add it. You can add to search list by adding the path to CMake list variable `CMAKE_PREFIX_PATH`. 
ssss