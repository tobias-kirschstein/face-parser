from dataclasses import dataclass
from typing import List

import numpy as np
import pyfvvdp


@dataclass
class PairedVideoMetric:
    jod: float

    def __add__(self, other: 'PairedVideoMetric') -> 'PairedVideoMetric':
        return PairedVideoMetric(
            jod=self.jod + other.jod,
        )

    def __radd__(self, other) -> 'PairedVideoMetric':
        if other == 0:
            return self

        return self + other

    def __truediv__(self, scalar: float) -> 'PairedVideoMetric':
        return PairedVideoMetric(
            jod=self.jod / scalar,
        )


class PairedVideoEvaluator:

    def __init__(self, fps: float):
        self._jod_evaluator = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='threshold')
        self._evaluation_fps = fps

    def evaluate(self, prediction_video: List[np.ndarray], target_video: List[np.ndarray]) -> PairedVideoMetric:
        prediction_video = np.stack(prediction_video)  # [T, H, W, C]
        target_video = np.stack(target_video)  # [T, H, W, C]
        jod, _ = self._jod_evaluator.predict(prediction_video, target_video, dim_order="FHWC",
                                             frames_per_second=max(4.1, self._evaluation_fps))

        paired_video_metric = PairedVideoMetric(jod=jod.item())
        return paired_video_metric
