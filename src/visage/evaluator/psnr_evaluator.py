import torch
from elias.util.batch import batchify_sliced
from torch import nn
from torchmetrics import PeakSignalNoiseRatio


class PSNREvaluator(nn.Module):

    def __init__(self, batch_size: int = 16):
        super(PSNREvaluator, self).__init__()
        self._psnr_evaluator = PeakSignalNoiseRatio(data_range=1.0)
        self._batch_size = batch_size

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        psnrs = []
        for predictions_batch, targets_batch in zip(batchify_sliced(predictions, self._batch_size), batchify_sliced(targets, self._batch_size)):
            psnrs.append(self._psnr_evaluator(predictions_batch, targets_batch))

        psnr = torch.stack(psnrs).mean()
        return psnr
