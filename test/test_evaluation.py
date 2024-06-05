from unittest import TestCase

import numpy as np

from visage.evaluator.paired_face_image_evaluator import PairedFaceImageEvaluator
from visage.evaluator.paired_image_evaluator import PairedImageEvaluator


class EvaluationTest(TestCase):

    def test_paired_image_evaluator(self):
        paired_image_evaluator = PairedImageEvaluator()

        predictions = np.zeros((4, 512, 512, 3))
        targets = np.ones((4, 512, 512, 3))

        paired_image_metrics = paired_image_evaluator.evaluate(predictions, targets)

        print(paired_image_metrics)

    def test_paired_face_image_evaluator(self):
        paired_face_image_evaluator = PairedFaceImageEvaluator()

        predictions = np.zeros((4, 512, 512, 3))
        targets = np.ones((4, 512, 512, 3))

        paired_face_image_metrics = paired_face_image_evaluator.evaluate(predictions, targets)

        print(paired_face_image_metrics)
