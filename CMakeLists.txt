# Project
cmake_minimum_required(VERSION 3.8)
project(Spyker VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Options
option(ENABLE_NATIVE "Build for native CPU architecture" ON)
option(ENABLE_PYTHON "Build with Python module support" ON)
option(ENABLE_DNNL "Build the library with DNNL Support" OFF)
option(ENABLE_EXAMPLES "Build examples for the library" ON)
option(ENABLE_CUDA "Build the library with CUDA support" ON)
option(ENABLE_CUDNN "Build the library with CUDNN Support" ON)
option(CUDA_ARCH_LIST "List of CUDA Architectures to generate code for" "")
option(ENABLE_BLAS "Build the library with BLAS support" ON)
option(ENABLE_TESTS "Build tests for the library" ON)

set(NATIVE_ENABLED OFF)
set(PYTHON_ENABLED OFF)
set(CUDA_ENABLED OFF)
set(CUDNN_ENABLED OFF)
set(BLAS_ENABLED OFF)
set(DNNL_ENABLED OFF)

# Native Architecture
if(ENABLE_NATIVE)
    set(NATIVE_ARCH "-march=native")
    set(NATIVE_ENABLED ON)
endif()

# Python Interface
if(ENABLE_PYTHON)
    set(LIBRARY_TYPE "STATIC")
else()
    set(LIBRARY_TYPE "SHARED")
endif()

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 9)
            set(CMAKE_CUDA_STANDARD 14)
        else()
            set(CMAKE_CUDA_STANDARD 11)
        endif()
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if (NOT CUDA_ARCH_LIST)
            if(CUDAToolkit_VERSION_MAJOR EQUAL 6)
                set(CUDA_ARCH_LIST 20 30 32 35 50 52 53)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 7)
                set(CUDA_ARCH_LIST 20 30 32 35 50 52 53)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 8)
                set(CUDA_ARCH_LIST 20 30 32 35 50 52 53 60 61 62)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 9)
                set(CUDA_ARCH_LIST 30 32 35 37 50 52 53 60 61 62 70 72)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 10)
                set(CUDA_ARCH_LIST 30 32 35 37 50 52 53 60 61 62 70 72 75)
            elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11)
                set(CUDA_ARCH_LIST 35 37 50 52 53 60 61 62 70 72 75 80 86)
            endif()
        endif()

        set(CUDA_ENABLED ON)
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
        string(REPLACE ";" "-" TEMP_STRING "${CMAKE_CUDA_ARCHITECTURES}")
        set(DEFINE_CUDA SPYKER_USE_CUDA SPYKER_CUDA_ARCH=\"${TEMP_STRING}\")

        if(WIN32)
            set(LIBRARY_CUDA curand.lib cublas.lib)
        else()
            set(LIBRARY_CUDA culibos curand_static cublas_static)
        endif()

        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.1)
            if(WIN32)
                set(LIBRARY_CUDA ${LIBRARY_CUDA} cublasLt.lib)
            else()
                set(LIBRARY_CUDA ${LIBRARY_CUDA} cublasLt_static)
            endif()
        endif()
    endif()
endif()

if(CUDA_ENABLED AND ENABLE_CUDNN)
    find_package(CUDNN)

    if(CUDNN_FOUND)
        set(CUDNN_ENABLED ON)
        set(LIBRARY_CUDNN cudnn::cudnn)
        set(DEFINE_CUDNN SPYKER_USE_CUDNN SPYKER_CUDNN_VERSION=${CUDNN_VERSION_MAJOR})
    endif()
endif()

# BLAS Library
if(ENABLE_BLAS)
    add_subdirectory(3rd/blasw)
    set(BLAS_ENABLED ON)
    set(LIBRARY_BLAS BLASW::BLASW)
    set(DEFINE_BLAS SPYKER_USE_BLAS)
endif()

# DNNL Library
if(ENABLE_DNNL)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
    set(DNNL_ARCH_OPT_FLAGS "${NATIVE_ARCH}" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE CONVOLUTION MATMUL CACHE LIST "" FORCE)

    add_subdirectory(3rd/dnnl)
    set(DNNL_ENABLED ON)
    add_library(dnnl_interface INTERFACE IMPORTED)
    target_link_libraries(dnnl_interface INTERFACE dnnl)
    set(LIBRARY_DNNL dnnl_interface)
    set(DEFINE_DNNL SPYKER_USE_DNNL)
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
add_subdirectory(3rd/serio)
find_package(OpenMP REQUIRED)

# The Library
add_library(spyker ${LIBRARY_TYPE} ${HEADERS} ${SOURCES} ${CUDA_SOURCES})
target_include_directories(spyker PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/stb/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rd/half/include/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gen/>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/>")
target_link_libraries(spyker PUBLIC
    OpenMP::OpenMP_CXX)
target_link_libraries(spyker PRIVATE
    serio ${LIBRARY_BLAS} ${LIBRARY_CUDA} ${LIBRARY_DNNL} ${LIBRARY_CUDNN})
target_compile_definitions(spyker PUBLIC
    ${DEFINE_BLAS} ${DEFINE_CUDA} ${DEFINE_DNNL} ${DEFINE_CUDNN})
target_compile_options(spyker PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>: "${NATIVE_ARCH}" -fno-math-errno>)
set_target_properties(spyker PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

if(CUDA_ENABLED)
 set_target_properties(spyker PROPERTIES
     CUDA_SEPARABLE_COMPILATION ON)
endif()

# Python Interface
if(ENABLE_PYTHON)
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

# Summary
message(STATUS "")
message(STATUS "Summary:")
message(STATUS "Native Architectrure: ---- ${NATIVE_ENABLED}")
message(STATUS "Python Interface: -------- ${PYTHON_ENABLED}")
message(STATUS "CUDA: -------------------- ${CUDA_ENABLED}")
message(STATUS "CUDNN: ------------------- ${CUDNN_ENABLED}")
message(STATUS "BLAS: -------------------- ${BLAS_ENABLED}")
message(STATUS "DNNL: -------------------- ${DNNL_ENABLED}")
message(STATUS "")
