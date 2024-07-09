from unittest import TestCase

import numpy as np
from elias.util import load_img
from elias.util.io import resize_img, save_img
from visage.matting.background_matting_v2 import BackgroundMattingV2


class BackgroundMattingV2Test(TestCase):

    def test_background_matting_v2(self):
        model = BackgroundMattingV2()

        image = load_img(f"../images/tobi_cam_222200038.jpg")
        background_image = load_img("../images/tobi_bg_222200038.jpg")

        alpha_images = model.parse([image, image], [background_image, background_image])
        alpha_image = (alpha_images[0] * 255).astype(np.uint8)
        alpha_image_regression = load_img("../images/tobi_background_matting_v2.png")
        self.assertTrue((alpha_image == alpha_image_regression).all())
