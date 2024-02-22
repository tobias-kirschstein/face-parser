# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import inspect
from pathlib import Path

try:
    from .nms.cpu_nms import cpu_nms
except ModuleNotFoundError:
    # Hacky way to call the build script that creates the cpu_nms python files
    # Ideally, this would be a post-install script, but there does not seem to be a good way
    import os
    import subprocess
    import visage

    cwd = os.getcwd()
    new_wd = Path(inspect.getfile(visage)).parent.parent#.parent
    cmd = ["python", "face_parser/modules/FaceBoxesV2/utils/build.py", "build_ext", "--inplace"]
    os.chdir(new_wd)
    print(f"Running: {' '.join(cmd)}")
    print(f"From directory: {new_wd}")
    # subprocess.call(f"python {Path(__file__).parent}/build.py build_ext --inplace")
    subprocess.call(cmd)
    os.chdir(cwd)

    from .nms.cpu_nms import cpu_nms


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets, thresh)
