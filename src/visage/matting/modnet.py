from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from modnet.models.modnet import MODNet
import torch.nn.functional as F
class MODNetMatter:

    def __init__(self):
        ckpt_path = f"{Path.home()}/.cache/visage/MODNet/modnet_photographic_portrait_matting.ckpt"

        # define image to tensor transform
        self._transform = transforms.Compose(
            [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        self._modnet = MODNet(backbone_pretrained=False)
        self._modnet = nn.DataParallel(self._modnet)

        if torch.cuda.is_available():
            self._modnet = self._modnet.cuda()
            weights = torch.load(ckpt_path)
        else:
            weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self._modnet.load_state_dict(weights)
        self._modnet.eval()

    def parse(self, images: torch.Tensor) -> torch.Tensor:
        """

        :param images:
            Can be one or many images. Either in channel-first or channel-last format:
             - [B, 3, H, W]
             - [B, H, W, 3]
             - [3, H, W]
             - [H, W, 3]
            Images should be normalized to [0, 1]!
        :return:
            A matting mask of the same shape as the input, but the channel dimension collapsed.
            Example input: [B, 3, H, W] -> [B, H, W]
        """

        needs_squeeze = False
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            needs_squeeze = True

        if images.shape[-1] == 3:
            # torchvision transforms requires images in Channel-first format
            images = images.permute(0, 3, 1, 2)

        # inference images
        images = self._transform(images)
        ref_size = 512

        # resize image for input
        im_b, im_c, im_h, im_w = images.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        images = F.interpolate(images, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = self._modnet(images.cuda() if torch.cuda.is_available() else images, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte.squeeze(-3)

        if needs_squeeze:
            matte = matte.squeeze(0)

        return matte