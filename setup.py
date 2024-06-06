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


ext_modules = [
    Extension(
        "visage.modules.FaceBoxesV2.utils.nms.cpu_nms",
        ["src/visage/modules/FaceBoxesV2/utils/nms/cpu_nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"] if sys.platform == 'linux' else None,
    )
]

if __name__ == "__main__":
    setuptools.setup(
        include_package_data=True,
        # Important to also install .c/.cu/.hpp/.pyx files for nms
        # And .txt for meanfaces
        package_data={'': ['*.c', '*.pyx', '*.hpp', '*.cu', '*.txt']},

        # For building NMS module
        ext_modules=ext_modules,
        cmdclass={'build_ext': custom_build_ext},
        setup_requires=['numpy'],
    )
