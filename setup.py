#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        include_package_data=True,
        # Important to also install .c/.cu/.hpp/.pyx files for nms
        # And .txt for meanfaces
        package_data={'': ['*.c', '*.pyx', '*.hpp', '*.cu', '*.txt']}, )
