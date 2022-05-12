set(HELPER_PATHS
    "${CUDNN_ROOT_DIR}"
    "$ENV{CUDNN_PATH}"
    "/opt/cudnn/"
    "/opt/cudnn6/"
    "/opt/cudnn7/"
    "/opt/cudnn8/")

find_file(VERSION_FILE
    NO_CACHE
    NAMES cudnn_version.h cudnn.h
    PATHS ${HELPER_PATHS}
    PATH_SUFFIXES include)

find_path(CUDNN_INCLUDE_DIRS
    NAMES cudnn.h
    PATHS ${HELPER_PATHS}
    PATH_SUFFIXES include)

if(NOT VERSION_FILE STREQUAL "VERSION_FILE-NOTFOUND")
    file(READ "${VERSION_FILE}" VERSION_TEXT)
    string(REGEX MATCH "CUDNN_MAJOR[\t ]*([0-9]+)" _ ${VERSION_TEXT})
    set(CUDNN_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "CUDNN_MINOR[\t ]*([0-9]+)" _ ${VERSION_TEXT})
    set(CUDNN_VERSION_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "CUDNN_PATCHLEVEL[\t ]*([0-9]+)" _ ${VERSION_TEXT})
    set(CUDNN_VERSION_PATCH ${CMAKE_MATCH_1})
    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
endif()

if(WIN32)
    set(LIBNAME_CNN cudnn_cnn_infer.lib)
    set(LIBNAME_OPS cudnn_ops_infer.lib)
    set(LIBNAME_OLD cudnn.lib)
else()
    set(LIBNAME_CNN cudnn_cnn_infer_static)
    set(LIBNAME_OPS cudnn_ops_infer_static)
    set(LIBNAME_OLD cudnn_static)
endif()

find_library(CUDNN_CNNLIB
    NAMES ${LIBNAME_CNN}
    PATHS ${HELPER_PATHS}
    PATH_SUFFIXES lib64 lib lib/x64 lib/x86)

find_library(CUDNN_OPSLIB
    NAMES ${LIBNAME_OPS}
    PATHS ${HELPER_PATHS}
    PATH_SUFFIXES lib64 lib lib/x64 lib/x86)

find_library(CUDNN_OLDLIB
    NAMES ${LIBNAME_OLD}
    PATHS ${HELPER_PATHS}
    PATH_SUFFIXES lib64 lib lib/x64 lib/x86)

if((NOT CUDNN_CNNLIB STREQUAL "CUDNN_CNNLIB-NOTFOUND") AND (NOT CUDNN_OPSLIB STREQUAL "CUDNN_OPSLIB-NOTFOUND"))
    set(CUDNN_LIBRARIES ${CUDNN_CNNLIB} ${CUDNN_OPSLIB})
else()
    set(CUDNN_LIBRARIES ${CUDNN_OLDLIB})
endif()

unset(LIBNAME_CNN)
unset(LIBNAME_OPS)
unset(LIBNAME_OLD)

unset(CUDNN_CNNLIB)
unset(CUDNN_OPSLIB)
unset(CUDNN_OLDLIB)

unset(HELPER_PATHS)
unset(VERSION_FILE)
unset(VERSION_TEXT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    FOUND_VAR CUDNN_FOUND
    REQUIRED_VARS
    CUDNN_LIBRARIES
    CUDNN_INCLUDE_DIRS
    VERSION_VAR CUDNN_VERSION)

#set(TEMP ${CMAKE_FIND_LIBRARY_SUFFIXES})
#SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#find_package(ZLIB REQUIRED)
#find_package(CUDAToolkit REQUIRED)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ${TEMP})

if(NOT WIN32)
    find_package(ZLIB REQUIRED)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARIES} ${ZLIB_LIBRARIES})
    set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS})
endif()

find_package(CUDAToolkit REQUIRED)

if(WIN32)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARIES} cublas.lib)
else()
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARIES} culibos cublas_static)
endif()

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 10.1)
    if(WIN32)
        set(CUDNN_LIBRARIES ${CUDNN_LIBRARIES} cublasLt.lib)
    else()
        set(CUDNN_LIBRARIES ${CUDNN_LIBRARIES} cublasLt_static)
    endif()
endif()

if(CUDNN_FOUND AND NOT TARGET cudnn::cudnn)
    add_library(cudnn::cudnn INTERFACE IMPORTED)
    set_target_properties(cudnn::cudnn PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${CUDNN_LIBRARIES}"
        VERSION "${CUDNN_VERSION}" SOVERSION "${CUDNN_VERSION_MAJOR}")
endif()
