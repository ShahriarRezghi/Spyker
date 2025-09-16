# BSD 3-Clause License
#
# Copyright (c) 2022-2025, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Project
cmake_minimum_required(VERSION 3.24)
project(Spyker VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_compile_options(-Wa,--noexecstack)
add_link_options(-Wl,-z,noexecstack)

# Options
set(SPYKER_OPTIM_FLAGS "-march=native" CACHE STRING "Optimization flags to build with")
set(SPYKER_CUDA_ARCH "" CACHE STRING "List of CUDA Architectures to generate code for")
option(SPYKER_ENABLE_PYTHON "Build with Python module support" OFF)
option(SPYKER_ENABLE_CUDA "Build the library with CUDA support" ON)
option(SPYKER_ENABLE_CUDNN "Build the library with CUDNN Support" ON)
option(SPYKER_ENABLE_DNNL "Build the library with DNNL Support" ON)
option(SPYKER_ENABLE_BLAS "Build the library with BLAS support" ON)
option(SPYKER_ENABLE_EXAMPLES "Build examples for the library" ON)
option(SPYKER_ENABLE_TESTS "Build tests for the library" OFF)
# External Options:
#  BLASW_FORCE_MKL: BOOL -> Force MKL for BLAS Backend
#  BLASW_BACKEND_ROOT: PATH -> Root for BLAS or MKL Backend
#  BLASW_BACKEND_STATIC: BOOL -> Static Library for BLAS Backend
#  BLASW_BACKEND_PROVIDER: STRING -> Provider for MKL or BLAS Backend
#  CMAKE_INSTALL_PREFIX: PATH -> Set the installation directory

set(PYTHON_ENABLED OFF)
set(CUDA_ENABLED OFF)
set(CUDNN_ENABLED OFF)
set(BLAS_ENABLED OFF)
set(DNNL_ENABLED OFF)

# Python Interface
if(SPYKER_ENABLE_PYTHON)
    set(LIBRARY_TYPE "STATIC")
else()
    set(LIBRARY_TYPE "SHARED")
endif()

# BLAS Library
if(SPYKER_ENABLE_BLAS)
    add_subdirectory(3rd/blasw)

    set(BLAS_ENABLED ON)
    set(LIBRARY_BLAS BLASW::BLASW)
    set(DEFINE_BLAS SPYKER_USE_BLAS)

    if(CBLAS_MKL)
        set(DEFINE_BLAS ${DEFINE_BLAS} SPYKER_USE_MKL)
    endif()
endif()

# DNNL Library
if(SPYKER_ENABLE_DNNL)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
    set(DNNL_ARCH_OPT_FLAGS "${SPYKER_OPTIM_FLAGS}" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE "CONVOLUTION;MATMUL" CACHE STRING "" FORCE)

    add_subdirectory(3rd/dnnl)
    set(DNNL_ENABLED ON)
    add_library(dnnl_interface INTERFACE IMPORTED)
    target_link_libraries(dnnl_interface INTERFACE dnnl)
    set(LIBRARY_DNNL dnnl_interface)

    file(READ "${CMAKE_CURRENT_BINARY_DIR}/3rd/dnnl/include/oneapi/dnnl/dnnl_version.h" dnnl_version_header)
    string(REGEX MATCH "#define DNNL_VERSION_MAJOR[ \t]+[0-9]+" _dnnl_major_line "${dnnl_version_header}")
    string(REGEX MATCH "[0-9]+" DNNL_VERSION_MAJOR "${_dnnl_major_line}")
    string(REGEX MATCH "#define DNNL_VERSION_MINOR[ \t]+[0-9]+" _dnnl_minor_line "${dnnl_version_header}")
    string(REGEX MATCH "[0-9]+" DNNL_VERSION_MINOR "${_dnnl_minor_line}")
    string(REGEX MATCH "#define DNNL_VERSION_PATCH[ \t]+[0-9]+" _dnnl_patch_line "${dnnl_version_header}")
    string(REGEX MATCH "[0-9]+" DNNL_VERSION_PATCH "${_dnnl_patch_line}")

    set(DEFINE_DNNL SPYKER_USE_DNNL
        SPYKER_DNNL_MAJOR=${DNNL_VERSION_MAJOR}
        SPYKER_DNNL_MINOR=${DNNL_VERSION_MINOR}
        SPYKER_DNNL_PATCH=${DNNL_VERSION_PATCH})
endif()

string(REPLACE " " ";" SPYKER_OPTIM_FLAGS "${SPYKER_OPTIM_FLAGS}")
string(REPLACE " " ";" SPYKER_CUDA_ARCH "${SPYKER_CUDA_ARCH}")

# CUDA
if(SPYKER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
            set(CMAKE_CUDA_STANDARD 17)
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 9)
            set(CMAKE_CUDA_STANDARD 14)
        else()
            set(CMAKE_CUDA_STANDARD 11)
        endif()

        if (NOT SPYKER_CUDA_ARCH)
            if(CUDAToolkit_VERSION_MAJOR EQUAL 6)
                set(SPYKER_CUDA_ARCH 20 30 32 35 50 52 53)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 7)
                set(SPYKER_CUDA_ARCH 20 30 32 35 50 52 53)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 8)
                set(SPYKER_CUDA_ARCH 20 30 32 35 50 52 53 60 61 62)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 9)
                set(SPYKER_CUDA_ARCH 30 32 35 37 50 52 53 60 61 62 70 72)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 10)
                set(SPYKER_CUDA_ARCH 30 32 35 37 50 52 53 60 61 62 70 72 75)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11)
                set(SPYKER_CUDA_ARCH 35 37 50 52 53 60 61 62 70 72 75 80 86)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12)
                if(CUDAToolkit_VERSION_MINOR GREATER_EQUAL 9)
                    set(SPYKER_CUDA_ARCH 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 103 120 121)
                elseif(CUDAToolkit_VERSION_MINOR GREATER_EQUAL 8)
                    set(SPYKER_CUDA_ARCH 50 52 53 60 61 62 70 72 75 80 86 87 89 90 100 103 120)
                else()
                    set(SPYKER_CUDA_ARCH 50 52 53 60 61 62 70 72 75 80 86 87 89 90)
                endif()
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
                set(SPYKER_CUDA_ARCH 75 80 86 87 89 90 100 103 120 121)
            endif()
        endif()

        set(CUDA_ENABLED ON)
        set(CMAKE_CUDA_ARCHITECTURES ${SPYKER_CUDA_ARCH})
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
        set(LIBRARY_CUDA CUDA::cudart_static CUDA::curand_static CUDA::cublas_static)

        string(REPLACE ";" "-" TEMP_STRING "${CMAKE_CUDA_ARCHITECTURES}")
        set(DEFINE_CUDA SPYKER_USE_CUDA
            SPYKER_CUDA_ARCH=\"${TEMP_STRING}\"
            SPYKER_CUDA_MAJOR=${CUDAToolkit_VERSION_MAJOR}
            SPYKER_CUDA_MINOR=${CUDAToolkit_VERSION_MINOR}
            SPYKER_CUDA_PATCH=${CUDAToolkit_VERSION_PATCH})

        if(NOT WIN32)
            set(LIBRARY_CUDA ${LIBRARY_CUDA} CUDA::culibos)
        endif()

        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.1)
            set(LIBRARY_CUDA ${LIBRARY_CUDA} CUDA::cublasLt_static)
        endif()

        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.0 AND NOT WIN32)
            set(LIBRARY_CUDA ${LIBRARY_CUDA} CUDA::nvrtc_static)
        endif()
    endif()
endif()

# cuDNN
if(CUDA_ENABLED AND SPYKER_ENABLE_CUDNN)
    find_package(CUDNN)

    if(CUDNN_FOUND)
        set(CUDNN_ENABLED ON)
        set(LIBRARY_CUDNN CUDNN::cudnn_needed)
        set(DEFINE_CUDNN SPYKER_USE_CUDNN
            SPYKER_CUDNN_MAJOR=${CUDNN_VERSION_MAJOR}
            SPYKER_CUDNN_MINOR=${CUDNN_VERSION_MINOR}
            SPYKER_CUDNN_PATCH=${CUDNN_VERSION_PATCH})

        if(CUDNN_VERSION_MAJOR GREATER_EQUAL 9)
            set(LIBRARY_CUDA $<LINK_LIBRARY:WHOLE_ARCHIVE, ${LIBRARY_CUDA}>)
        endif()
    endif()
endif()

# Project Files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gen/spyker/config.h")
file(GLOB HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/cpu/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/sparse/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/helper/*.h")
file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/cpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/sparse/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/helper/*.cpp")
file(GLOB CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/spyker/cuda/*")
file(GLOB PYTHON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bind/*.cpp")

# Dependencies
find_package(OpenMP REQUIRED CXX)

# The Library
add_library(spyker ${LIBRARY_TYPE} ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
target_include_directories(spyker PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/stb>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/stb/deprecated>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/half/include/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gen/>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/>")
target_link_libraries(spyker PUBLIC
    OpenMP::OpenMP_CXX)
target_link_libraries(spyker PRIVATE
    ${LIBRARY_BLAS} ${LIBRARY_DNNL} ${LIBRARY_CUDNN} ${LIBRARY_CUDA})
target_compile_definitions(spyker PUBLIC
    ${DEFINE_BLAS} ${DEFINE_CUDA} ${DEFINE_DNNL} ${DEFINE_CUDNN})
target_compile_options(spyker PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>: ${SPYKER_OPTIM_FLAGS} -fno-math-errno -ftree-vectorize>)
set_target_properties(spyker PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

if(CUDA_ENABLED)
    set_target_properties(spyker PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

# Python Interface
if(SPYKER_ENABLE_PYTHON)
    add_subdirectory(3rd/pybind11)
    set(PYTHON_ENABLED ON)
    add_library(spyker_plugin MODULE ${PYTHON_SOURCES})
    set_target_properties(spyker_plugin PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}")
    target_link_libraries(spyker_plugin PRIVATE
        spyker pybind11::module)
endif()

# Examples
if(SPYKER_ENABLE_EXAMPLES)
    add_executable(play play/play.cpp)
    target_link_libraries(play PUBLIC spyker)
endif()

# Summary
message(STATUS "")
message(STATUS "Summary:")
message(STATUS "Optimization Flags: ------------ ${SPYKER_OPTIM_FLAGS}")
message(STATUS "CUDA Architecture List: -------- ${SPYKER_CUDA_ARCH}")
message(STATUS "Python Interface: -------------- ${PYTHON_ENABLED}")
message(STATUS "CUDA: -------------------------- ${CUDA_ENABLED}")
message(STATUS "CUDNN: ------------------------- ${CUDNN_ENABLED}")
message(STATUS "DNNL: -------------------------- ${DNNL_ENABLED}")
message(STATUS "BLAS: -------------------------- ${BLAS_ENABLED}")
message(STATUS "")
