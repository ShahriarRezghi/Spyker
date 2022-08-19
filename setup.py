import os
import re
import sys
import platform
import subprocess
import multiprocessing

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


def IsTrue(arg):
    if arg == 1: return True
    if arg.upper() == 'ON': return True
    if arg.upper() == 'TRUE': return True


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DENABLE_PYTHON=ON', '-DENABLE_TESTS=OFF',
                      '-DENABLE_EXAMPLES=OFF', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]
        env = os.environ.copy()

        if 'ENABLE_NINJA' in env and IsTrue(env['ENABLE_NINJA']):
            cmake_args.append('-GNinja')
        if 'CCACHE_EXEC' in env:
            cmake_args.append('-DCMAKE_CXX_COMPILER_LAUNCHER=' + env['CCACHE_EXEC'])
        if 'CMAKE_ARGS' in env:
            cmake_args.extend(env['CMAKE_ARGS'].split())
        if 'OPTIM_FLAGS' in env:
            cmake_args.append('-DOPTIM_FLAGS=' + env['OPTIM_FLAGS'])

        if 'BLA_STATIC' in env:
            cmake_args.append('-DBLA_STATIC=' + env['BLA_STATIC'])
        if 'BLA_VENDOR' in env:
            cmake_args.append('-DBLA_VENDOR=' + env['BLA_VENDOR'])
        if 'CUDNN_ROOT_DIR' in env:
            cmake_args.append('-DCUDNN_ROOT_DIR=' + env['CUDNN_ROOT_DIR'])
        if 'CUDA_ARCH_LIST' in env:
            cmake_args.append('-DCUDA_ARCH_LIST=' + env['CUDA_ARCH_LIST'])

        if 'ENABLE_BLAS' in env:
            cmake_args.append('-DENABLE_BLAS=' + env['ENABLE_BLAS'])
        if 'ENABLE_DNNL' in env:
            cmake_args.append('-DENABLE_DNNL=' + env['ENABLE_DNNL'])
        if 'ENABLE_CUDA' in env:
            cmake_args.append('-DENABLE_CUDA=' + env['ENABLE_CUDA'])
        if 'ENABLE_CUDNN' in env:
            cmake_args.append('-DENABLE_CUDNN=' + env['ENABLE_CUDNN'])

        build_type = 'Release'
        if 'BUILD_TYPE' in env:build_type = env['BUILD_TYPE']
        cmake_args.append('-DCMAKE_BUILD_TYPE=' + build_type)

        build_args = []
        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{build_type.upper()}={extdir}']
            if sys.maxsize > 2 ** 32: cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', f'-j{multiprocessing.cpu_count()}']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(['cmake', ext.sourcedir] + cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        subprocess.check_call(['rm', '-f', 'libmkldnn.a'], cwd=extdir)

setup(
    name='spyker',
    version='0.1.0',
    author='Shahriar Rezghi',
    author_email='shahriar25.ss@gmail.com',
    description='Spiking Deep Neural Networks Library',
    packages=find_packages('src/python'),
    package_dir={'': 'src/python'},
    ext_modules=[CMakeExtension('spyker/spyker_plugin')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
