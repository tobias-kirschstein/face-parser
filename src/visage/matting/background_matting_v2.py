from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Union, Literal, List

import numpy as np
import requests
import torch
from BackgroundMattingV2.model import MattingBase, MattingRefine
from elias.util import ensure_directory_exists_for_file
from torch import nn
from elias.config import Config


@dataclass
class BackgroundMattingV2Config(Config):
    model_type: Literal['mattingbase', 'mattingrefine'] = 'mattingrefine'
    model_backbone: Literal['mobilenetv2', 'resnet50', 'resnet101'] = 'resnet101'
    model_backbone_scale: float = 0.25
    model_refine_mode: Literal['full', 'sampling', 'thresholding'] = 'thresholding'
    model_refine_sample_pixels: int = 80000
    model_refine_threshold: float = 0.01
    model_refine_kernel_size: int = 3


class BackgroundMattingV2:

    def __init__(self,
                 config: BackgroundMattingV2Config = BackgroundMattingV2Config(),
                 device: torch.device = torch.device('cuda')):

        self._config = config
        self._device = device

        checkpoint_path = f"{Path().home()}/.cache/torch/face-parser/BackgroundMattingV2/pytorch_resnet101.pth"
        if not Path(checkpoint_path).exists():
            download_path = "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/pytorch_resnet101.pth"
            if not Path(checkpoint_path).exists():
                print(f"Downloading {download_path} into {checkpoint_path}")
                request = requests.get(download_path, allow_redirects=True)
                ensure_directory_exists_for_file(checkpoint_path)
                open(checkpoint_path, 'wb').write(request.content)

        self._model = self._setup_model(
            checkpoint_path, config.model_type, config.model_backbone, config.model_backbone_scale,
            config.model_refine_mode, config.model_refine_sample_pixels, config.model_refine_threshold,
            config.model_refine_kernel_size, device
        )

    def _setup_model(
            self,
            model_checkpoint: str,
            model_type: Literal['mattingbase', 'mattingrefine'],
            model_backbone: Literal['mobilenetv2', 'resnet50', 'resnet101'],
            model_backbone_scale: float,
            model_refine_mode: Literal['full', 'sampling', 'thresholding'],
            model_refine_sample_pixels: int,
            model_refine_threshold: float,
            model_refine_kernel_size: int,
            device: torch.device,
    ) -> Union[MattingBase, MattingRefine]:

        print('Initializing model...')

        # Load model
        if model_type == 'mattingbase':
            model = MattingBase(model_backbone)
        if model_type == 'mattingrefine':
            model = MattingRefine(
                model_backbone,
                model_backbone_scale,
                model_refine_mode,
                model_refine_sample_pixels,
                model_refine_threshold,
                model_refine_kernel_size)

        model = model.to(device).eval()
        model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)

        return model

    def parse(self, images: List[np.ndarray], background_images: List[np.ndarray]) -> List[np.ndarray]:

        alpha_images = []
        prepared_images = []
        prepared_background_images = []
        original_sizes = []
        for image, background_image in zip(images, background_images):
            H, W, C = image.shape
            original_size = (W, H)
            enlarged_size = (int(ceil((W / 4)) * 4), int(ceil((H / 4)) * 4))
            original_sizes.append(original_size)

            src_enlarged = np.zeros((enlarged_size[1], enlarged_size[0], C), dtype=image.dtype)
            src_enlarged[:original_size[1], :original_size[0]] = image
            bgr_enlarged = np.zeros((enlarged_size[1], enlarged_size[0], C), dtype=background_image.dtype)
            bgr_enlarged[:original_size[1], :original_size[0]] = background_image

            src = (torch.tensor(src_enlarged).permute(2, 0, 1).float() / 255).to(self._device, non_blocking=True)
            bgr = (torch.tensor(bgr_enlarged).permute(2, 0, 1).float() / 255).to(self._device, non_blocking=True)

            prepared_images.append(src)
            prepared_background_images.append(bgr)

        prepared_images = torch.stack(prepared_images)
        prepared_background_images = torch.stack(prepared_background_images)

        with torch.no_grad():
            if self._config.model_type == 'mattingbase':
                pha, fgr, err, _ = self._model(prepared_images, prepared_background_images)
            elif self._config.model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = self._model(prepared_images, prepared_background_images)

                for i, original_size in enumerate(original_sizes):
                    pha[i] = pha[i, :, :original_size[1], :original_size[0]]
                    # fgr[i] = fgr[i, :, :original_size[1], :original_size[0]]
                    # err[i] = err[i, :, :original_size[1], :original_size[0]]
            else:
                raise ValueError(f"Unknown model_type: {self._config.model_type}")

        alpha_images = pha[:, 0].cpu().numpy()
        # fgr = fgr.cpu()
        # err = err.cpu()

        return alpha_images

