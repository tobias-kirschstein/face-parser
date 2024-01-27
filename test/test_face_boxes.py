import cv2
from matplotlib import pyplot as plt

from base import VisionTestCase
from face_parser.bounding_boxes.face_boxes_v2 import FaceBoxesV2


class FaceBoxTest(VisionTestCase):

    def test_face_box(self):
        img = self._get_obama_image(
            normalize=False,
            channel_first=False,
            keep_aspect_ratio=True,
            resize=None)
        img = (img * 255).int().numpy()

        detector = FaceBoxesV2()
        # detector = detector.cuda()
        my_thresh = 0.6

        detections = detector.detect(img, my_thresh)
        print(detections)

        cv2.rectangle(img,
                      detections[0].get_point1(),
                      detections[0].get_point2(),
                      (255, 0, 0),
                      10)

        plt.imshow(img)
        plt.show()
