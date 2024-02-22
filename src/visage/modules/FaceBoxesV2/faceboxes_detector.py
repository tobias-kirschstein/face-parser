import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import requests
import torch
from torch import nn

from visage.modules.FaceBoxesV2.detector import Detector
from visage.modules.FaceBoxesV2.utils.box_utils import decode
from visage.modules.FaceBoxesV2.utils.config import cfg
from visage.modules.FaceBoxesV2.utils.faceboxes import FaceBoxesV2Internal
from visage.modules.FaceBoxesV2.utils.nms_wrapper import nms
from visage.modules.FaceBoxesV2.utils.prior_box import PriorBox


@dataclass
class DetectedBBox:
    x: int
    y: int
    width: int
    height: int
    score: float

    def get_point1(self) -> Tuple[int, int]:
        return self.x, self.y

    def get_point2(self) -> Tuple[int, int]:
        return self.x + self.width, self.y + self.height


class FaceBoxesDetector(Detector):
    def __init__(self):
        ckpt_folder = f"{Path().home()}/.cache/torch/face-parser"
        ckpt_path = f"{ckpt_folder}/FaceBoxesV2.pth"
        os.makedirs(ckpt_folder, exist_ok=True)
        # Download path from PIPNet repository:
        # https://github.com/jhb86253817/PIPNet/blob/master/FaceBoxesV2/weights/FaceBoxesV2.pth
        download_path = "https://github.com/jhb86253817/PIPNet/raw/master/FaceBoxesV2/weights/FaceBoxesV2.pth"

        if not Path(ckpt_path).exists():
            print(f"Downloading {download_path} into {ckpt_path}")
            request = requests.get(download_path, allow_redirects=True)
            open(ckpt_path, 'wb').write(request.content)

        super().__init__('FaceBoxes', ckpt_path)

        self.name = 'FaceBoxesDetector'
        self.net = FaceBoxesV2Internal(phase='test', size=None, num_classes=2)  # initialize detector
        state_dict = torch.load(self.model_weights, map_location=torch.device('cpu'))
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.net.load_state_dict(new_state_dict)
        self.net.eval()

        self.device_indicator_param = nn.Parameter()

    def detect(self, image: np.ndarray, thresh: float = 0.6) -> List[DetectedBBox]:

        device = self.device_indicator_param
        image_scale = image

        scale = torch.Tensor([image_scale.shape[1], image_scale.shape[0], image_scale.shape[1], image_scale.shape[0]])
        image_scale = torch.from_numpy(image_scale.transpose(2, 0, 1)).to(device).int()
        mean_tmp = torch.IntTensor([104, 117, 123]).to(device).int()
        mean_tmp = mean_tmp.unsqueeze(1).unsqueeze(2)
        image_scale -= mean_tmp
        image_scale = image_scale.float().unsqueeze(0)
        scale = scale.to(device)

        with torch.no_grad():
            out = self.net(image_scale)
            # priorbox = PriorBox(cfg, out[2], (image_scale.size()[2], image_scale.size()[3]), phase='test')
            priorbox = PriorBox(cfg, image_size=(image_scale.size()[2], image_scale.size()[3]))
            priors = priorbox.forward()
            priors = priors.to(device)
            loc, conf = out
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > thresh)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(dets, 0.3)
            dets = dets[keep, :]

            dets = dets[:750, :]
            detections_scale = []
            for i in range(dets.shape[0]):
                xmin = int(dets[i][0])
                ymin = int(dets[i][1])
                xmax = int(dets[i][2])
                ymax = int(dets[i][3])
                score = dets[i][4]
                width = xmax - xmin
                height = ymax - ymin
                detections_scale.append(DetectedBBox(xmin, ymin, width, height, score))

        return detections_scale
