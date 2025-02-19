from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
import torch

from visage.evaluator.csim_evaluator import CSIMEvaluator
from visage.evaluator.keypoint_evaluator import KeypointEvaluator
from visage.evaluator.paired_image_evaluator import PairedImageMetrics, PairedImageEvaluator


@dataclass
class PairedFaceImageMetrics(PairedImageMetrics):
    akd: Optional[float] = None
    akd_face: Optional[float] = None
    csim: Optional[float] = None
    aed: Optional[float] = None
    apd: Optional[float] = None

    def __add__(self, other: 'PairedFaceImageMetrics') -> 'PairedFaceImageMetrics':
        paired_image_metrics = super().__add__(other)
        return PairedFaceImageMetrics(
            **asdict(paired_image_metrics),
            akd=self.akd + other.akd if self.akd is not None and other.akd is not None else None,
            akd_face=self.akd_face + other.akd_face if self.akd_face is not None and other.akd_face is not None else None,
            csim=self.csim + other.csim if self.csim is not None and other.csim is not None else None,
            aed=self.aed + other.aed if self.aed is not None and other.aed is not None else None,
            apd=self.apd + other.apd if self.apd is not None and other.apd is not None else None,
        )

    def __radd__(self, other) -> 'PairedFaceImageMetrics':
        if other == 0:
            return self

        return self + other

    def __truediv__(self, scalar: float) -> 'PairedFaceImageMetrics':
        paired_image_metrics = super().__truediv__(scalar)
        return PairedFaceImageMetrics(
            **asdict(paired_image_metrics),
            akd=self.akd / scalar if self.akd is not None else None,
            akd_face=self.akd_face / scalar if self.akd_face is not None else None,
            csim=self.csim / scalar if self.csim is not None else None,
            aed=self.aed / scalar if self.aed is not None else None,
            apd=self.apd / scalar if self.apd is not None else None,
        )

    def __mul__(self, scalar: float) -> 'PairedFaceImageMetrics':
        paired_image_metrics = super().__truediv__(scalar)
        return PairedFaceImageMetrics(
            **asdict(paired_image_metrics),
            akd=self.akd * scalar if self.akd is not None else None,
            akd_face=self.akd_face * scalar if self.akd_face is not None else None,
            csim=self.csim * scalar if self.csim is not None else None,
            aed=self.aed * scalar if self.aed is not None else None,
            apd=self.apd * scalar if self.apd is not None else None,
        )


class PairedFaceImageEvaluator(PairedImageEvaluator):

    def __init__(self, exclude_lpips: bool = False, exclude_mssim: bool = False):
        super().__init__(exclude_lpips=exclude_lpips, exclude_mssim=exclude_mssim)
        self._keypoint_evaluator = KeypointEvaluator()
        self._csim_evaluator = CSIMEvaluator()

    def evaluate(self, predictions: List[np.ndarray], targets: List[np.ndarray]) -> PairedFaceImageMetrics:
        return super().evaluate(predictions, targets)

    def _evaluate(self, predictions_torch: torch.Tensor, targets_torch: torch.Tensor) -> PairedFaceImageMetrics:
        paired_image_metrics = super()._evaluate(predictions_torch, targets_torch)

        akd, akd_face = self._keypoint_evaluator(predictions_torch, targets_torch)
        csim = self._csim_evaluator(predictions_torch, targets_torch)

        return PairedFaceImageMetrics(**asdict(paired_image_metrics), akd=akd.item(), akd_face=akd_face.item(), csim=csim.item())
