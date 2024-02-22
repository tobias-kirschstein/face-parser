import torch
from matplotlib import pyplot as plt

from base import VisionTestCase
from visage.bisenet import BiSeNetFaceParser
from visage.visualize import apply_colormap


class BiSeNetTest(VisionTestCase):

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
        image = self._get_obama_image(keep_aspect_ratio=True, resize=512)

        with torch.no_grad():
            face_parser = BiSeNetFaceParser()

            segmentation_mask = face_parser.parse(image)
            colormap = apply_colormap(segmentation_mask)

            self.assertEqual(colormap.shape[0], image.shape[1])
            self.assertEqual(colormap.shape[1], image.shape[2])
            self.assertEqual(colormap.shape[2], image.shape[0])

            plt.imshow(0.5 * colormap / 255 + 0.5 * (image.permute(1, 2, 0).numpy() + 1) / 2)
            plt.show()
