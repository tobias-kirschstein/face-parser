from typing import Dict

import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101


class DeepLabV3FaceParser:

    def __init__(self):
        seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to('cuda:0')
        seg_net.requires_grad_(False)
        seg_net.eval()

        self._seg_net = seg_net
        self._transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
            A segmentation mask of the same shape as the input, but the channel dimension collapsed
            which indicates 1 out of 20 classes for each pixel.
            Example input: [B, 3, H, W] -> [B, H, W]
        """

        needs_squeeze = False
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
            needs_squeeze = True

        if images.shape[-1] == 3:
            # torchvision transforms requires images in Channel-first format
            images = images.permute(0, 3, 1, 2)

        images = self._transform(images)
        images = images.cuda()
        output = self._seg_net(images)['out']
        segmentation_masks = torch.nn.functional.softmax(output, dim=1)
        segmentation_masks = segmentation_masks.cpu().detach().argmax(1)

        if needs_squeeze:
            segmentation_masks = segmentation_masks.squeeze(0)

        return segmentation_masks

    @classmethod
    def get_label_mapping(cls) -> Dict[int, str]:

        return {0: 'background',
                1: 'aeroplane',
                2: 'bicycle',
                3: 'bird',
                4: 'boat',
                5: 'bottle',
                6: 'bus',
                7: 'car',
                8: 'cat',
                9: 'chair',
                10: 'cow',
                11: 'diningtable',
                12: 'dog',
                13: 'horse',
                14: 'motorbike',
                15: 'person',
                16: 'pottedplant',
                17: 'sheep',
                18: 'sofa',
                19: 'train',
                20: 'tvmonitor'}
