import inspect
from pathlib import Path
from typing import Optional
from unittest import TestCase

import torch
from PIL import Image
from torchvision.transforms import PILToTensor

import face_parser

IMAGE_URL_OBAMA = "https://upload.wikimedia.org/wikipedia/commons/f/f9/Obama_portrait_crop.jpg"


class VisionTestCase(TestCase):

    def _get_obama_image(self,
                         channel_first: bool = True,
                         normalize: bool = True,
                         resize: Optional[int] = None,
                         keep_aspect_ratio: bool = True) -> torch.Tensor:
        pil_img = Image.open(f"{Path(inspect.getfile(face_parser)).parent.parent.parent}/images/obama.jpg")

        if resize is not None:
            if keep_aspect_ratio:
                w = pil_img.size[0]
                h = pil_img.size[1]
                aspect_ratio = w / h

                if w > h:
                    new_w = resize
                    new_h = int(resize / aspect_ratio)
                else:
                    new_w = int(resize * aspect_ratio)
                    new_h = resize
            else:
                new_w = resize
                new_h = resize
            pil_img = pil_img.resize((new_w, new_h))

        img = PILToTensor()(pil_img)

        if normalize:
            img = img / 255 * 2 - 1
        else:
            img = img / 255

        if not channel_first:
            img = img.permute(1, 2, 0)

        return img
