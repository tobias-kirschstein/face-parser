import torch
from elias.util.batch import batchify_sliced
from torch import nn
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure


class SSIMEvaluator(nn.Module):

    def __init__(self, batch_size: int = 16):
        super(SSIMEvaluator, self).__init__()
        self._ssim_evaluator = StructuralSimilarityIndexMeasure(data_range=1.0)
        self._batch_size = batch_size

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ssims = []
        for predictions_batch, targets_batch in zip(batchify_sliced(predictions, self._batch_size), batchify_sliced(targets, self._batch_size)):
            ssims.append(self._ssim_evaluator(predictions_batch, targets_batch))

        ssim = torch.stack(ssims).mean()
        return ssim


class MultiScaleSSIMEvaluator(nn.Module):

    def __init__(self, batch_size: int = 16):
        super(MultiScaleSSIMEvaluator, self).__init__()
        self._batch_size = batch_size
        self._multi_scale_ssim_evaluator = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        multi_scale_ssims = []
        for predictions_batch, targets_batch in zip(batchify_sliced(predictions, self._batch_size), batchify_sliced(targets, self._batch_size)):
            multi_scale_ssims.append(self._multi_scale_ssim_evaluator(predictions_batch, targets_batch))

        multi_scale_ssim = torch.stack(multi_scale_ssims).mean()
        return multi_scale_ssim
