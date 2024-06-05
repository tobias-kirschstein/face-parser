from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from dreifus.image import Img
from matplotlib.cm import get_cmap
from torch.nn import MSELoss

from visage.evaluator.lpips_evaluator import LPIPSEvaluator
from visage.evaluator.psnr_evaluator import PSNREvaluator
from visage.evaluator.ssim_evaluator import SSIMEvaluator, MultiScaleSSIMEvaluator


@dataclass
class PairedImageMetrics:
    psnr: float
    ssim: float
    lpips: float
    multi_scale_ssim: Optional[float] = None
    mse: Optional[float] = None

    def __add__(self, other: 'PairedImageMetrics') -> 'PairedImageMetrics':
        return PairedImageMetrics(
            psnr=self.psnr + other.psnr,
            ssim=self.ssim + other.ssim,
            lpips=self.lpips + other.lpips,
            multi_scale_ssim=self.multi_scale_ssim + other.multi_scale_ssim if self.multi_scale_ssim is not None and other.multi_scale_ssim is not None else None,
            mse=self.mse + other.mse if self.mse is not None and other.mse is not None else None,
        )

    def __radd__(self, other) -> 'PairedImageMetrics':
        if other == 0:
            return self

        return self + other

    def __truediv__(self, scalar: float) -> 'PairedImageMetrics':
        return PairedImageMetrics(
            psnr=self.psnr / scalar,
            ssim=self.ssim / scalar,
            lpips=self.lpips / scalar,
            multi_scale_ssim=self.multi_scale_ssim / scalar if self.multi_scale_ssim is not None else None,
            mse=self.mse / scalar if self.mse is not None else None,
        )


class PairedImageEvaluator:

    def __init__(self, exclude_lpips: bool = False):
        self._psnr_evaluator = PSNREvaluator()
        self._ssim_evaluator = SSIMEvaluator()
        self._multi_scale_ssim_evaluator = MultiScaleSSIMEvaluator()
        if not exclude_lpips:
            self._lpips_evaluator = LPIPSEvaluator()
        self._mse_evaluator = MSELoss()

        self._exclude_lpips = exclude_lpips

    def evaluate(self, predictions: List[np.ndarray], targets: List[np.ndarray]) -> PairedImageMetrics:
        predictions_torch = [Img.from_numpy(prediction).to_torch().img for prediction in predictions]
        targets_torch = [Img.from_numpy(target).to_torch().img for target in targets]

        # Exclude targets that are all 0s
        proper_target_mask = [~((target == 0) | (target == 128)).all(axis=-1).all(axis=-1).all(axis=-1)
                              for target in targets]
        predictions_torch = [prediction for i, prediction in enumerate(predictions_torch) if proper_target_mask[i]]
        targets_torch = [target for i, target in enumerate(targets_torch) if proper_target_mask[i]]

        predictions_torch = torch.stack(predictions_torch)  # [B, 3, H, W]
        targets_torch = torch.stack(targets_torch)  # [B, 3, H, W]

        assert predictions_torch.min() >= 0
        assert predictions_torch.max() <= 1
        assert targets_torch.max() <= 1
        assert targets_torch.min() >= 0

        return self._evaluate(predictions_torch, targets_torch)

    def _evaluate(self, predictions_torch: torch.Tensor, targets_torch: torch.Tensor) -> PairedImageMetrics:
        psnr = self._psnr_evaluator(predictions_torch, targets_torch)
        ssim = self._ssim_evaluator(predictions_torch, targets_torch)
        multi_scale_ssim = self._multi_scale_ssim_evaluator(predictions_torch, targets_torch)
        if self._exclude_lpips:
            lpips = torch.tensor(-1)
        else:
            lpips = self._lpips_evaluator(predictions_torch, targets_torch)
        mse = self._mse_evaluator(predictions_torch, targets_torch)

        return PairedImageMetrics(psnr=psnr.item(), ssim=ssim.item(), lpips=lpips.item(), multi_scale_ssim=multi_scale_ssim.item(), mse=mse.item())

    def compute_error_maps(self, predictions: List[np.ndarray], targets: List[np.ndarray]) -> List[np.ndarray]:
        error_maps = []
        for prediction, target in zip(predictions, targets):
            error = np.linalg.norm(prediction / 255. - target / 255., axis=-1, ord=1)
            cmap = get_cmap('turbo')
            error_map = cmap(error)
            error_maps.append(error_map[..., :3])

        return error_maps
