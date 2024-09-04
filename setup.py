#!/usr/bin/env python
import sys
from distutils.extension import Extension

import setuptools
from Cython.Distutils import build_ext  # Note, this requires Cython to be listed under "required" in pyproject.toml
from setuptools.command.build_ext import build_ext


# ----------------------------------------------------------
# Build procedure for compiled NMS module of FaceBoxesV2
# ----------------------------------------------------------

# run the customize_compiler
class custom_build_ext(build_ext):

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # __builtins__.__NUMPY_SETUP__ = False
        # Obtain the numpy include directory.  This logic works across numpy versions.
        import numpy as np

        try:
            numpy_include = np.get_include()
        except AttributeError:
            numpy_include = np.get_numpy_include()

        self.include_dirs.append(numpy_include)

    def build_extensions(self):
        # customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



if __name__ == "__main__":
    # This is a very hacky way to implement a Python version switch for Cython
    # The underlying problem is that in Python 3.9 onwards, numpy>=2 is available which deprecated the use of np.int_t
    # However, in Python 3.8 with numpy<2, the replacement np.int64_t does not work as there are differences between Windows and Linux
    # The nice solution would be to add a simple compiler directive in the .pyx file which checks for the Python version. This seems to be not so simple.
    # Therefore, we copied the whole file, replaced the lines in question, and then select already in setup.py which file should be compiled
    if sys.version_info >= (3, 9):
        cpu_nms_path = "src/visage/modules/FaceBoxesV2/utils/nms/cpu_nms_py39.pyx"
    else:
        cpu_nms_path = "src/visage/modules/FaceBoxesV2/utils/nms/cpu_nms.pyx"

    setuptools.setup(
        include_package_data=True,
        # Important to also install .c/.cu/.hpp/.pyx files for nms
        # And .txt for meanfaces
        package_data={'': ['*.c', '*.pyx', '*.hpp', '*.cu', '*.txt']},

        # For building NMS module
        ext_modules=[
            Extension(
                "visage.modules.FaceBoxesV2.utils.nms.cpu_nms",
                [cpu_nms_path],
                extra_compile_args=["-Wno-cpp", "-Wno-unused-function"] if sys.platform == 'linux' else None,
            )
        ],
        cmdclass={'build_ext': custom_build_ext},
        setup_requires=['numpy'],
    )
