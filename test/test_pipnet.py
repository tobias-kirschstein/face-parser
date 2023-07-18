import cv2
from matplotlib import pyplot as plt

from base import VisionTestCase
from face_parser.bounding_boxes.face_boxes_v2 import FaceBoxesV2
from face_parser.landmark_detection.pipnet import PIPNet


class PIPNetTest(VisionTestCase):

    def test_pipnet(self):
        pip_net = PIPNet()
        pip_net = pip_net.cuda()

        facebox_detector = FaceBoxesV2()
        facebox_detector = facebox_detector.cuda()

        img = self._get_obama_image(
            normalize=False,
            channel_first=False,
            keep_aspect_ratio=True,
            resize=None)
        img = (img * 255).int().numpy()

        detected_bboxes = facebox_detector.detect(img)
        landmarks = pip_net.forward(img, detected_bboxes[0])

        for x, y in landmarks:
            cv2.circle(img, (int(x), int(y)), 10, (255, 0, 0), -1)

        plt.imshow(img)
        plt.show()
