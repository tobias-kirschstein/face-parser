from pathlib import Path
from unittest import TestCase

import torch
from PIL import Image
from torchvision import transforms

from face_parser.bisenet import BiSeNetFaceParser
from face_parser.visualize import apply_colormap


class BiSeNetTest(TestCase):

    def test_parse(self):
        face_parser = BiSeNetFaceParser()

        B = 5
        H = 71
        W = 83

        images = torch.randn((B, H, W, 3))
        segmentation_masks = face_parser.parse(images)

        self.assertEqual(len(segmentation_masks.shape), 3)
        self.assertEqual(segmentation_masks.shape[0], B)
        self.assertEqual(segmentation_masks.shape[1], H)
        self.assertEqual(segmentation_masks.shape[2], W)

    def test_real_image(self):
        image = Image.open(f"{Path(__file__).parent.resolve()}/../test_img.png")
        image = transforms.ToTensor()(image)

        with torch.no_grad():
            face_parser = BiSeNetFaceParser()

            # to_tensor = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # ])
            #
            # img = to_tensor(image)
            # img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            # out = face_parser._model(img)[0]
            # segmentation_mask = out.squeeze(0).cpu().numpy().argmax(0)

            segmentation_mask = face_parser.parse(image)
            colormap = apply_colormap(segmentation_mask)

            self.assertEqual(colormap.shape[0], image.shape[1])
            self.assertEqual(colormap.shape[1], image.shape[2])
            self.assertEqual(colormap.shape[2], image.shape[0])

