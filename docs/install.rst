Spyker Installation & Setup Guide
=================================

.. contents::
   :local:
   :depth: 2

This document describes how to build and install Spyker from source, configure optional
accelerators, and prepare the Python bindings. The instructions consolidate the build-time
options exposed by the top-level ``CMakeLists.txt``, the Python ``setup.py`` wrapper, and helper
projects such as ``3rd/blasw``.

Prerequisites
-------------

General
~~~~~~~

- **CMake ≥ 3.24** (the top-level project requires it; ``setup.py`` verifies the version).
- A **C++11** compiler (GCC/Clang/MSVC) with OpenMP support.
- ``git`` and a POSIX shell (examples assume Linux/macOS; translate commands for PowerShell on
  Windows).
- Build tools for native dependencies: ``make`` or ``ninja`` (optional but recommended).

Math Backends
~~~~~~~~~~~~~

Spyker can optionally use external BLAS, cuDNN, and oneDNN (DNNL).

- **BLAS**: Required when ``SPYKER_ENABLE_BLAS`` is ``ON`` (default). You may point CMake at an
  existing BLAS/LAPACK installation or let the bundled ``3rd/blasw`` helper manage discovery. MKL,
  OpenBLAS, FlexiBLAS, and Apple Accelerate are supported.
- **oneDNN (DNNL)**: Enabled by default (`SPYKER_ENABLE_DNNL=ON`) for optimized CPU convolution/
  matmul. The build fetches and builds a static oneDNN unless you override the dependency.
- **CUDA + cuDNN**: Spyker ships with CUDA kernels and a cuDNN integration. CUDA support defaults
  to ``ON`` (``SPYKER_ENABLE_CUDA=ON``). cuDNN becomes active automatically when both CUDA and
  ``SPYKER_ENABLE_CUDNN=ON`` succeed.

Required System Packages
~~~~~~~~~~~~~~~~~~~~~~~~

Depending on your platform you may need to install the following packages yourself:

- GNU build toolchain (``build-essential`` on Debian/Ubuntu, ``xcode-select --install`` on macOS).
- CUDA Toolkit and driver (from NVIDIA) if building with CUDA.
- cuDNN (download from NVIDIA or install via package managers) and expose headers/libs through
  the environment variables documented below.
- ``zlib`` development headers (cuDNN discovery links against ZLIB).
- Python 3.9-3.14 and ``pip`` if you plan to build the wheels.

Source Tree Layout
------------------

Key directories referenced in this guide:

- ``src/`` - C++ sources and headers for the core library and CUDA kernels.
- ``src/python`` - Python package (pure-Python helpers and pybind11 module glue).
- ``3rd/`` - vendored third-party components (BLASW dispatcher, oneDNN, pybind11, stb, etc.).
- ``cmake/`` - helper CMake modules (``FindCUDNN.cmake`` and exported package config).

Configuring The C++ Build
-------------------------

The root ``CMakeLists.txt`` surface the following cache variables and options:

+------------------------------+----------------------------------------------------------------+-------------------+
| Variable / Option            | Purpose                                                        | Default           |
+==============================+================================================================+===================+
| ``SPYKER_OPTIM_FLAGS``       | Extra compiler flags (space-separated).                        | ``-march=native`` |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_CUDA_ARCH``         | CUDA SM list (space-separated). Overrides auto-detection.      | auto              |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_PYTHON``     | Build pybind11 module. Forces static ``libspyker``.            | ``OFF``           |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_CUDA``       | Compile CUDA backend.                                          | ``ON``            |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_CUDNN``      | Link against cuDNN (requires CUDA).                            | ``ON``            |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_DNNL``       | Build and link oneDNN.                                         | ``ON``            |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_BLAS``       | Enable BLAS (via BLASW helper).                                | ``ON``            |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_EXAMPLES``   | Build ``play`` demo executable.                                | ``ON``            |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``SPYKER_ENABLE_TESTS``      | Build tests (none shipped by default).                         | ``OFF``           |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``BLASW_BACKEND_ROOT``       | Root for BLAS/MKL backend; seeds ``BLAS_ROOT``/``MKL_ROOT``.   | *(empty)*         |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``BLASW_BACKEND_STATIC``     | Prefer static (``ON``) or dynamic (``OFF``) backend linking.   | *(empty)*         |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``BLASW_BACKEND_PROVIDER``   | Select BLAS vendor or MKL triple.                              | *(empty)*         |
+------------------------------+----------------------------------------------------------------+-------------------+
| ``BLASW_FORCE_MKL``          | Force Intel MKL backend regardless of detection.               | ``OFF``           |
+------------------------------+----------------------------------------------------------------+-------------------+

You can set these via ``-DNAME=value`` on the CMake command line or by exporting environment
variables that ``setup.py`` forwards (see *Python Build Environment Variables*).

Example: CPU-only Release build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cmake -S . -B build \
     -DSPYKER_ENABLE_CUDA=OFF \
     -DSPYKER_ENABLE_CUDNN=OFF \
     -DSPYKER_ENABLE_DNNL=ON \
     -DSPYKER_OPTIM_FLAGS="-O3 -march=native"
   cmake --build build -j$(nproc)
   cmake --install build --prefix /opt/spyker

Example: CUDA build with explicit SM targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cmake -S . -B build \
     -DSPYKER_ENABLE_CUDA=ON \
     -DSPYKER_CUDA_ARCH="80 86" \
     -DSPYKER_ENABLE_CUDNN=ON
   cmake --build build --target spyker -j$(nproc)

Environment Variables for cuDNN Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FindCUDNN.cmake`` honours the following variables when locating headers and libraries:

- ``CUDNN_INCLUDE_PATH`` or ``CUDNN_PATH``: root directory containing ``include/cudnn*.h``.
- ``CUDNN_LIBRARY_PATH`` or ``CUDNN_PATH``: directories containing ``libcudnn_*`` libraries.
- ``Python_SITEARCH``: searched to support ``pip install nvidia-cudnn-cu11`` style layouts.

Ensure ``ZLIB`` development files are discoverable; ``FindCUDNN.cmake`` links ``ZLIB::ZLIB`` into
``CUDNN::cudnn_needed``.

BLAS Backend Configuration (3rd/blasw)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BLASW helper accepts granular configuration through the ``BLASW_BACKEND_*`` cache entries
summarised in the table above. These complement the standard BLAS/MKL variables (``BLA_VENDOR``,
``BLA_STATIC``, ``MKL_ROOT``, ``MKL_INTERFACE``, ``MKL_THREADING``, etc.), which remain available
for advanced setups.

Building the Python Package
---------------------------

Spyker's Python wheel is built via ``setuptools`` but delegates all compilation to CMake. Typical
invocations:

Install in the current environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install .

Editable install for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install --editable .

Both commands run CMake in a temporary build directory and honour the environment variables
summarised below.

Python Build Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``setup.py`` forwards a curated set of variables to CMake:

+----------------------------+------------------------------------------------------------------------------+
| Environment Variable       | Effect                                                                       |
+============================+==============================================================================+
| ``SPYKER_ENABLE_NINJA``    | Generate Ninja build files when set to ``ON``/``TRUE``/``1``.                |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_CCACHE_EXEC``     | Path to a compiler launcher (e.g. ``ccache``) forwarded as ``-DCMAKE_CXX_COMPILER_LAUNCHER``. |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_CMAKE_ARGS``      | Additional space-separated CMake flags appended verbatim.                    |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_OPTIM_FLAGS``     | Overrides the ``SPYKER_OPTIM_FLAGS`` cache entry.                            |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_CUDA_ARCH``       | Overrides automatic CUDA SM detection.                                       |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_BUILD_TYPE``      | Sets ``CMAKE_BUILD_TYPE`` (defaults to ``Release``).                         |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_ENABLE_BLAS``     | Enable or disable the BLAS backend.                                          |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_ENABLE_DNNL``     | Enable or disable oneDNN integration.                                        |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_ENABLE_CUDA``     | Toggle CUDA compilation.                                                     |
+----------------------------+------------------------------------------------------------------------------+
| ``SPYKER_ENABLE_CUDNN``    | Toggle cuDNN linkage.                                                        |
+----------------------------+------------------------------------------------------------------------------+
| ``BLASW_FORCE_MKL``        | Forwarded as ``-DBLASW_FORCE_MKL`` to force Intel MKL.                       |
+----------------------------+------------------------------------------------------------------------------+
| ``BLASW_BLAS_ROOT``        | Forwarded as ``-DBLASW_BACKEND_ROOT`` to seed BLAS/MKL paths.                |
+----------------------------+------------------------------------------------------------------------------+
| ``BLASW_BLAS_STATIC``      | Forwarded as ``-DBLASW_BACKEND_STATIC`` to request static or dynamic linking. |
+----------------------------+------------------------------------------------------------------------------+
| ``BLASW_BLAS_PROVIDER``    | Forwarded as ``-DBLASW_BACKEND_PROVIDER`` to select the BLAS vendor or MKL triple. |
+----------------------------+------------------------------------------------------------------------------+

The build process uses all available CPU cores by default (``-j`` flag). On Windows the script also
selects the x64 generator automatically when running on a 64-bit interpreter.

Output Artifacts
~~~~~~~~~~~~~~~~

- ``spyker/spyker_plugin`` - compiled pybind11 module (installed into the Python package).
- ``libspyker`` - shared or static C++ library (static when ``SPYKER_ENABLE_PYTHON=ON``).
- Optional ``play`` example executable if ``SPYKER_ENABLE_EXAMPLES=ON``.

Troubleshooting & Tips
----------------------

- Inspect the CMake *Summary* banner at the end of configuration to confirm which backends are
  active (CUDA/CUDNN/DNNL/BLAS and optimization flags).
- When linking against custom cuDNN builds, double-check that both headers and libraries are
  reachable via the documented environment variables.
- If you see missing BLAS symbols, reconfigure with ``-DBLASW_BACKEND_PROVIDER=OpenBLAS`` (or similar)
  and point ``BLASW_BACKEND_ROOT`` at the installation root.
- Set ``SPYKER_ENABLE_EXAMPLES=OFF`` on headless build servers to avoid linking the sample binary.
- Clear the ``build`` directory between major configuration changes to avoid stale cache entries.

Verification
------------

After installation, verify the library and Python bindings:

.. code-block:: bash

   python - <<'PY'
   import spyker
   print("Spyker version:", spyker.version())
   print("Devices:", spyker.all_devices())
   PY

If CUDA is enabled you can further validate with ``spyker.cuda_available()`` and inspect memory
statistics via ``spyker.cuda_memory_total()``.

Next Steps
----------

- Explore the high-level usage examples in ``USAGE.rst``.
- Build the Sphinx documentation (``sphinx-build docs build/html``) for API details.
- Review ``docs/install.rst`` for platform-specific notes or package manager instructions once
  available.
