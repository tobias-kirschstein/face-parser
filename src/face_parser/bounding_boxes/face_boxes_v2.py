from typing import List

import numpy as np

from face_parser.modules.FaceBoxesV2.faceboxes_detector import FaceBoxesDetector, DetectedBBox


class FaceBoxesV2(FaceBoxesDetector):

    def forward(self, image: np.ndarray, thresh: float = 0.6) -> List[DetectedBBox]:
        return self.detect(image, thresh=thresh)
