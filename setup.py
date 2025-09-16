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

import multiprocessing
import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def is_true(arg):
    if arg == 1:
        return True
    if arg.upper() == "ON":
        return True
    if arg.upper() == "TRUE":
        return True


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DSPYKER_ENABLE_PYTHON=ON",
            "-DSPYKER_ENABLE_TESTS=OFF",
            "-DSPYKER_ENABLE_EXAMPLES=OFF",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPython_EXECUTABLE=" + sys.executable,
            "-DPython3_EXECUTABLE=" + sys.executable,
        ]
        env = os.environ.copy()

        if "SPYKER_ENABLE_NINJA" in env and is_true(env["SPYKER_ENABLE_NINJA"]):
            cmake_args.append("-GNinja")
        if "SPYKER_CCACHE_EXEC" in env:
            cmake_args.append("-DCMAKE_CXX_COMPILER_LAUNCHER=" + env["SPYKER_CCACHE_EXEC"])
        if "SPYKER_CMAKE_ARGS" in env:
            cmake_args.extend(env["SPYKER_CMAKE_ARGS"].split())
        if "SPYKER_OPTIM_FLAGS" in env:
            cmake_args.append("-DSPYKER_OPTIM_FLAGS=" + env["SPYKER_OPTIM_FLAGS"])

        if "BLASW_FORCE_MKL" in env:
            cmake_args.append("-DBLASW_FORCE_MKL=" + env["BLASW_FORCE_MKL"])
        if "BLASW_BLAS_ROOT" in env:
            cmake_args.append("-DBLASW_BACKEND_ROOT=" + env["BLASW_BLAS_ROOT"])
        if "BLASW_BLAS_STATIC" in env:
            cmake_args.append("-DBLASW_BACKEND_STATIC=" + env["BLASW_BLAS_STATIC"])
        if "BLASW_BLAS_PROVIDER" in env:
            cmake_args.append("-DBLASW_BACKEND_PROVIDER=" + env["BLASW_BLAS_PROVIDER"])
        if "SPYKER_CUDA_ARCH" in env:
            cmake_args.append("-DSPYKER_CUDA_ARCH=" + env["SPYKER_CUDA_ARCH"])

        if "SPYKER_ENABLE_BLAS" in env:
            cmake_args.append("-DSPYKER_ENABLE_BLAS=" + env["SPYKER_ENABLE_BLAS"])
        if "SPYKER_ENABLE_DNNL" in env:
            cmake_args.append("-DSPYKER_ENABLE_DNNL=" + env["SPYKER_ENABLE_DNNL"])
        if "SPYKER_ENABLE_CUDA" in env:
            cmake_args.append("-DSPYKER_ENABLE_CUDA=" + env["SPYKER_ENABLE_CUDA"])
        if "SPYKER_ENABLE_CUDNN" in env:
            cmake_args.append("-DSPYKER_ENABLE_CUDNN=" + env["SPYKER_ENABLE_CUDNN"])

        build_type = "Release"
        if "SPYKER_BUILD_TYPE" in env:
            build_type = env["SPYKER_BUILD_TYPE"]
        cmake_args.append("-DCMAKE_BUILD_TYPE=" + build_type)

        build_args = []
        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{build_type.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", f"-j{multiprocessing.cpu_count()}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(["cmake", ext.sourcedir] + cmake_args)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)
        subprocess.check_call(["rm", "-f", "libmkldnn.a"], cwd=extdir)


setup(
    packages=find_packages("src/python"),
    package_dir={"": "src/python"},
    ext_modules=[CMakeExtension("spyker/spyker_plugin")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
