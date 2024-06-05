from typing import Tuple

import numpy as np
import torch
from dreifus.image import Img
from visage.bounding_boxes.face_boxes_v2 import FaceBoxesV2
from visage.landmark_detection.pipnet import PIPNet


class KeypointEvaluator:

    def __init__(self):
        self._detector = FaceBoxesV2()
        self._pip_net = PIPNet()

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keypoint_distances = []
        keypoint_distances_face = []
        for prediction, target in zip(predictions, targets):
            prediction = Img.from_torch(prediction).to_numpy().img
            target = Img.from_torch(target).to_numpy().img
            detected_bboxes_prediction = self._detector(prediction)
            detected_bboxes_target = self._detector(target)
            if len(detected_bboxes_prediction) > 0 and len(detected_bboxes_target) > 0:
                landmarks_prediction = self._pip_net.forward(prediction, detected_bboxes_prediction[0])
                landmarks_target = self._pip_net.forward(target, detected_bboxes_target[0])

                keypoint_distance = np.linalg.norm(landmarks_prediction - landmarks_target, axis=-1).mean()
                keypoint_distance_face = np.linalg.norm(landmarks_prediction[33:] - landmarks_target[33:], axis=-1).mean()
                keypoint_distances.append(keypoint_distance)
                keypoint_distances_face.append(keypoint_distance_face)

        if len(keypoint_distances) > 0:
            average_keypoint_distance = torch.tensor(np.stack(keypoint_distances).mean())
            average_keypoint_distance_face = torch.tensor(np.stack(keypoint_distances_face).mean())
        else:
            average_keypoint_distance = torch.tensor(-1)
            average_keypoint_distance_face = torch.tensor(-1)

        return average_keypoint_distance, average_keypoint_distance_face
