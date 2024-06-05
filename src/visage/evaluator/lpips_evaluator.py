import torch
from elias.util.batch import batchify_sliced
from torch import nn
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity


class LPIPSEvaluator(nn.Module):

    def __init__(self, batch_size: int = 16):
        super(LPIPSEvaluator, self).__init__()
        self._batch_size = batch_size
        self._lpips_evaluator = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lpipses = []
        for predictions_batch, targets_batch in zip(batchify_sliced(predictions, self._batch_size), batchify_sliced(targets, self._batch_size)):
            lpipses.append(self._lpips_evaluator(predictions_batch, targets_batch))

        lpips = torch.stack(lpipses).mean()
        return lpips
