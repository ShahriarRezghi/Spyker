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

add_library(CUDNN::cudnn_needed INTERFACE IMPORTED)

find_path(
    CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_INCLUDE_PATH} ${CUDNN_INCLUDE_PATH} $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include
    REQUIRED
)

if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" cudnn_version_header)
    string(REGEX MATCH "#define CUDNN_MAJOR[ \t]+[0-9]+" _cudnn_major_line "${cudnn_version_header}")
    string(REGEX MATCH "[0-9]+" CUDNN_VERSION_MAJOR "${_cudnn_major_line}")
    string(REGEX MATCH "#define CUDNN_MINOR[ \t]+[0-9]+" _cudnn_minor_line "${cudnn_version_header}")
    string(REGEX MATCH "[0-9]+" CUDNN_VERSION_MINOR "${_cudnn_minor_line}")
    string(REGEX MATCH "#define CUDNN_PATCHLEVEL[ \t]+[0-9]+" _cudnn_patch_line "${cudnn_version_header}")
    string(REGEX MATCH "[0-9]+" CUDNN_VERSION_PATCH "${_cudnn_patch_line}")

else() # Fallback: parse cudnn.h
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" cudnn_header)
    string(REGEX MATCH "#define CUDNN_MAJOR[ \t]+[0-9]+" _cudnn_major_line "${cudnn_header}")
    string(REGEX MATCH "[0-9]+" CUDNN_VERSION_MAJOR "${_cudnn_major_line}")
    string(REGEX MATCH "#define CUDNN_MINOR[ \t]+[0-9]+" _cudnn_minor_line "${cudnn_header}")
    string(REGEX MATCH "[0-9]+" CUDNN_VERSION_MINOR "${_cudnn_minor_line}")
    string(REGEX MATCH "#define CUDNN_PATCHLEVEL[ \t]+[0-9]+" _cudnn_patch_line "${cudnn_header}")
    string(REGEX MATCH "[0-9]+" CUDNN_VERSION_PATCH "${_cudnn_patch_line}")
endif()

function(find_cudnn_library NAME)
    find_library(
        ${NAME}_LIBRARY ${NAME}_static
        HINTS $ENV{CUDNN_LIBRARY_PATH} ${CUDNN_LIBRARY_PATH} $ENV{CUDNN_PATH} ${CUDNN_PATH} ${Python_SITEARCH}/nvidia/cudnn ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib64 lib/x64 lib
        REQUIRED
    )

    if(${NAME}_LIBRARY)
        add_library(CUDNN::${NAME} UNKNOWN IMPORTED)
        set_target_properties(
            CUDNN::${NAME} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR}
            IMPORTED_LOCATION ${${NAME}_LIBRARY}
        )
        message(STATUS "${NAME} found at ${${NAME}_LIBRARY}.")
    else()
        message(STATUS "${NAME} not found.")
    endif()


endfunction()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    CUDNN REQUIRED_VARS
    CUDNN_INCLUDE_DIR
)

if(CUDNN_INCLUDE_DIR)
    message(STATUS "cuDNN: ${CUDNN_INCLUDE_DIR}")
    set(CUDNN_FOUND ON CACHE INTERNAL "cuDNN Library Found")
else()
    set(CUDNN_FOUND OFF CACHE INTERNAL "cuDNN Library Not Found")
endif()

find_package(ZLIB REQUIRED)
target_link_libraries(
    CUDNN::cudnn_needed
    INTERFACE
    ZLIB::ZLIB
)

target_include_directories(
    CUDNN::cudnn_needed
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>
)

if(CUDNN_VERSION_MAJOR EQUAL 8)
    find_cudnn_library(cudnn_adv_infer)
    find_cudnn_library(cudnn_adv_train)
    find_cudnn_library(cudnn_cnn_infer)
    find_cudnn_library(cudnn_cnn_train)
    find_cudnn_library(cudnn_ops_infer)
    find_cudnn_library(cudnn_ops_train)

    target_link_libraries(
        CUDNN::cudnn_needed
        INTERFACE
        CUDNN::cudnn_cnn_infer
        CUDNN::cudnn_ops_infer
    )
elseif(CUDNN_VERSION_MAJOR EQUAL 9)
    find_cudnn_library(cudnn_cnn)
    find_cudnn_library(cudnn_adv)
    find_cudnn_library(cudnn_graph)
    find_cudnn_library(cudnn_ops)
    find_cudnn_library(cudnn_engines_runtime_compiled)
    find_cudnn_library(cudnn_engines_precompiled)
    find_cudnn_library(cudnn_heuristic)

    target_link_libraries(
        CUDNN::cudnn_needed
        INTERFACE
        $<LINK_LIBRARY:WHOLE_ARCHIVE,
            CUDNN::cudnn_ops
            CUDNN::cudnn_cnn
            CUDNN::cudnn_graph
            CUDNN::cudnn_engines_precompiled
            CUDNN::cudnn_engines_runtime_compiled
        >
    )
endif()
