====================
Building From Source
====================

In order to build Spyker from source you will need to install some prequisites, get the source code, compile the library, and install it.

Prequisites
===========

There are some libraries Spyker depends upon that need to be installed on your system. All of these prequisites are optional but recommended to get the best performance out of the library. Most of them can be installed using linux package managers.

CUDA
----

In order to enable CUDA support in Spyker, you need to have CUDA on your system. Please refer to CUDA instalation guide `Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ and `Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_ for instructions on how to install it. CUDA version 6.5 or later is supported.

cuDNN
-----

Installing cuDNN is highly recommended as it gives the performance boost needed when running networks on GPUs. Please refer to `cuDNN installation guide <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`_ for instructions on how to install it.

CBLAS and LAPACKE
-----------------

Spyker does some linear algebra operations using CBLAS and LAPACKE libraries. Having a good implementation of these libraries can improve performance in the operations that use them. Some good options are  `Intel MKL <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html>`_ library on Intel CPUs and `OpenBLAS <https://www.openblas.net/>`_ (usually needs to be built from source to support LAPACKE).

Others
------

Spyker uses auto-vectorization to improve performance on CPUs. You will need a good and new C++ compiler to build Spyker. Some options are Clang (recommended) and GCC. Use the latest version if you can. You will also need a recent version of CMake so that it will have proper support for CUDA language.

Arch Linux
----------

Installing the prequisites is very easy in Arch linux using its package manager. For example:

.. code-block:: bash

    pacman -S base-devel clang cmake intel-mkl intel-mkl-static cuda cudnn

You might need sudo to run this command.

Build and Install
=================

First, you need to download the source code:

.. code-block:: bash

    git clone --recursive git@github.com:Mathific/Spyker.git
    cd spyker

Then, depending on which interface you want to build, you can begin the building process.

C++ Interface
-------------
The C++ interface can be built like so:

.. code-block:: bash

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j 8
    make install

You might need sudo to run ``make install``.

Python Interface
----------------

You can build the Python interface with:

.. code-block:: bash

    pip wheel -vvv . -w dist/


You will need to install the ``wheel`` package for Python. The built package is located in dist dictory and can be installed with ``pip install <package-file-name>``.

Build Options
-------------

There are some build options to give you control over the installation:

======================= ============== ====================== ===================================================================================================
Option                  Interface      Type                   Description
======================= ============== ====================== ===================================================================================================
ENABLE_TESTS            C++                                   Build the library tests
ENABLE_EXAMPLES         C++            CMake                  Build the library examples
CMAKE_INSTALL_PREFIX    C++            CMake                  Set the installation directory
ENABLE_DNNL             C++, Python    CMake                  Build the library with DNNL support (highly recommended)
ENABLE_BLAS             C++, Python    CMake(C++), Env(Py)    Build the library using CLBAS and LAPACKE support
ENABLE_CUDA             C++, Python    CMake(C++), Env(Py)    build the library using CUDA support
ENABLE_CUDNN            C++, Python    CMake(C++), Env(Py)    build the library using cuDNN support (highly recommended)
ENABLE_NATIVE           C++, Python    CMake(C++), Env(Py)    Build the library using native CPU instructions (must be off when building portable library)
ENABLE_NINJA            Python         CMake(C++), Env(Py)    Build the library with Ninja (pass ``-G Ninja`` for C++)
CUDA_ARCH_LIST          C++, Python    Env(Py)                CUDA architectures to be built for
MKLROOT                 C++, Python    CMake(C++), Env(Py)    MKL root directory hint path
CUDA_PATH               C++, Python    CMake(C++), Env(Py)    CUDA root directory hint path
BLA_VENDOR              C++, Python    CMake(C++), Env(Py)    CBLAS and LAPACKE vendor (`see this <https://cmake.org/cmake/help/latest/module/FindBLAS.html>`_)

======================= ============== ====================== ===================================================================================================

An example of setting a CMake variable:

.. code-block:: bash

    cmake -DVARIABLE:VALUE ...


An example of setting an environment variable:

.. code-block:: bash

    VARIABLE=VALUE pip ...


or by using ``export`` in shells.

Finding Dependencies
--------------------

finding dependencies can be tricky when it comes to a multi-platform library. Some tips are shown here that will help you find missing dependencies. So please consider these tips before opening an issue. The standard way of setting the hint paths for dependencies is listed in the ``Building Options`` section.

CMake package finding works by searching a list of paths for headers and library files. By default, this search is done quietly. You can enable verbose mode by setting ``CMAKE_FIND_DEBUG_MODE`` CMake option to ``ON``. You can check if the path that contains the needed files exists and if not, add it. You can add to search list by adding the path to CMake list variable ``CMAKE_PREFIX_PATH``.
