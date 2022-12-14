#!/usr/bin/env python
import os
from pathlib import Path
from subprocess import check_call
from sys import platform

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install


# TODO: blenderproc paths are hardcoded (cannot import blenderproc during setup to user their code...)
# TODO: Apparently the pip way of doing a post install hook is to run stuff in __init__.py the first time it is imported
# TODO: wrap post install with try-except and print instructions upon failure

def run_blenderproc_init(is_develop: bool = False):
    famudy_root_directory = Path(__file__).parent.resolve()

    print("Installing famudy package in blenderproc...")
    if platform == 'win32':
        blender_pip_path = [os.path.expanduser(
            "~/blender/blender-3.1.0-windows-x64/blender-3.1.0-windows-x64/3.1/python/bin/python.exe"), "-m", "pip"]
    else:
        check_call(["sudo", "apt-get", "install", "python3-dev"])
        blender_pip_path = [os.path.expanduser("~/blender/blender-3.1.0-linux-x64/3.1/python/bin/pip")]
    print("Assuming blender's pip is in ", blender_pip_path)

    blenderproc_cmd = blender_pip_path + \
                      ["install",
                       "-e" if is_develop else "",
                       str(famudy_root_directory).replace(" ",
                                                          "\\ ") if is_develop else "/home/tobias/Programming/Python/famudy",
                       '--install-option="--no-blenderproc-init"']
    print(" ".join(blenderproc_cmd))
    check_call(" ".join(blenderproc_cmd), shell=True)


def run_blenderproc_quickstart():
    # To download blender
    check_call(["blenderproc", "quickstart"])


# def get_blender_install_path():
#     # TODO: This should go into __init__.py
#     #   even then, the InstallUtility returns the wrong path. Hardcoding it for now...
#     import os
#     from argparse import Namespace
#     os.environ['OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT'] = '1'
#     from blenderproc.python.utility.InstallUtility import InstallUtility
#
#     args = Namespace(custom_blender_path=None, blender_install_path=None)
#     custom_blender_path, blender_install_path = InstallUtility.determine_blender_install_path(False, args, [])
#     blender_path, blender_version = InstallUtility.make_sure_blender_is_installed(custom_blender_path,
#                                                                                   blender_install_path)
#
#     print(blender_path)
#     blender_path = blender_path.replace("\\", "/")  # replace backslashes
#     return f"{blender_path}/{blender_version}"


class PostInstallCommand(install):
    user_options = install.user_options + [
        ('no-blenderproc-init', None, None),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.no_blenderproc_init = None

    def run(self):
        install.run(self)

        if not self.no_blenderproc_init:
            # run_blenderproc_quickstart()
            # get_blender_install_path()
            run_blenderproc_init()


class PostDevelopCommand(develop):
    user_options = develop.user_options + [
        ('no-blenderproc-init', None, None),
    ]

    def initialize_options(self):
        develop.initialize_options(self)
        self.no_blenderproc_init = None

    def run(self):
        develop.run(self)

        # TODO: Cannot do quickstart here. So probably have to force user to run blender quickstart manually
        if not self.no_blenderproc_init:
            # run_blenderproc_quickstart()
            # get_blender_install_path()
            run_blenderproc_init(is_develop=True)


if __name__ == "__main__":
    setuptools.setup(cmdclass={
        # 'develop': PostDevelopCommand,
        # 'install': PostInstallCommand
    })
