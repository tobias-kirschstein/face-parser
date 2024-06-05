from unittest import TestCase

import torch

from base import VisionTestCase
from visage.matting.modnet import MODNetMatter


class ModNetTest(VisionTestCase):
    def test_modnet(self):

        # define hyper-parameters
        ref_size = 512

        image = self._get_obama_image(keep_aspect_ratio=True, resize=688, normalize=False)
        # image = image[:, int((image.shape[1] - 512) / 2):512 + int((image.shape[1] - 512) / 2)]

        modnet_matter = MODNetMatter()
        matte = modnet_matter.parse(image)
        print('hi')