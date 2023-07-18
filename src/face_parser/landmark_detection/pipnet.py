import inspect
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision.transforms import transforms

import face_parser.modules.PIPNet
from face_parser.landmark_detection.pipnet_config import PIPNetConfig, PIPNet_WFLW_r18_config
from face_parser.modules.FaceBoxesV2.faceboxes_detector import DetectedBBox
from face_parser.modules.PIPNet.lib.functions import get_meanface, forward_pip
from face_parser.modules.PIPNet.lib.mobilenetv3 import mobilenetv3_large
from face_parser.modules.PIPNet.lib.networks import Pip_resnet18, Pip_resnet50, Pip_resnet101, Pip_mbnetv2, Pip_mbnetv3


class PIPNet(nn.Module):

    def __init__(self, config: PIPNetConfig = PIPNet_WFLW_r18_config):
        super(PIPNet, self).__init__()

        self._cfg = config
        self._det_box_scale = 1.2

        if self._cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=self._cfg.pretrained)
            net = Pip_resnet18(resnet18, self._cfg.num_nb, num_lms=self._cfg.num_lms, input_size=self._cfg.input_size,
                               net_stride=self._cfg.net_stride)
        elif self._cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=self._cfg.pretrained)
            net = Pip_resnet50(resnet50, self._cfg.num_nb, num_lms=self._cfg.num_lms, input_size=self._cfg.input_size,
                               net_stride=self._cfg.net_stride)
        elif self._cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=self._cfg.pretrained)
            net = Pip_resnet101(resnet101, self._cfg.num_nb, num_lms=self._cfg.num_lms, input_size=self._cfg.input_size,
                                net_stride=self._cfg.net_stride)
        elif self._cfg.backbone == 'mobilenet_v2':
            mbnet = models.mobilenet_v2(pretrained=self._cfg.pretrained)
            net = Pip_mbnetv2(mbnet, self._cfg.num_nb, num_lms=self._cfg.num_lms, input_size=self._cfg.input_size,
                              net_stride=self._cfg.net_stride)
        elif self._cfg.backbone == 'mobilenet_v3':
            mbnet = mobilenetv3_large()
            if self._cfg.pretrained:
                mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
            net = Pip_mbnetv3(mbnet, self._cfg.num_nb, num_lms=self._cfg.num_lms, input_size=self._cfg.input_size,
                              net_stride=self._cfg.net_stride)
        else:
            raise ValueError(f'Invalid backbone: {self._cfg.backbone}')

        weight_file = f"{Path().home()}/.cache/torch/face-parser/PIPNet/{self._cfg.checkpoint_path}"
        if not Path(weight_file).exists():
            raise ValueError(f"Download {self._cfg.checkpoint_path} from "
                             f"https://drive.google.com/drive/folders/1fz6UQR2TjGvQr4birwqVXusPp6tMAXxq and put it into "
                             f"{Path().home()}/.cache/torch/face-parser/PIPNet/")

        state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)

        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self._preprocess = transforms.Compose(
            [transforms.Resize((self._cfg.input_size, self._cfg.input_size)), transforms.ToTensor(), self._normalize])

        meanface_path = f"{Path(inspect.getfile(face_parser.modules.PIPNet)).parent}/meanfaces/{self._cfg.meanface_path}"
        meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(meanface_path, self._cfg.num_nb)

        self._meanface_indices = meanface_indices
        self._reverse_index1 = reverse_index1
        self._reverse_index2 = reverse_index2
        self._max_len = max_len

        net.eval()
        self._net = net
        self._device_indicator_param = nn.Parameter()

    def forward(self, image: np.ndarray, bbox: DetectedBBox) -> np.ndarray:
        device = self._device_indicator_param.device

        if bbox.score == -1:
            landmarks = None
        else:
            image_height, image_width, _ = image.shape
            det_xmin = int(bbox.x)
            det_ymin = int(bbox.y)
            det_width = int(bbox.width)
            det_height = int(bbox.height)
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (self._det_box_scale - 1) / 2)
            # remove a part of top area for alignment, see paper for details
            det_ymin += int(det_height * (self._det_box_scale - 1) / 2)
            det_xmax += int(det_width * (self._det_box_scale - 1) / 2)
            det_ymax += int(det_height * (self._det_box_scale - 1) / 2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width - 1)
            det_ymax = min(det_ymax, image_height - 1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :].astype(float)
            # plt.imshow(det_crop)
            # plt.show()
            det_crop = cv2.resize(det_crop, (self._cfg.input_size, self._cfg.input_size))
            inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
            inputs = self._preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(
                self._net,
                inputs,
                self._preprocess,
                self._cfg.input_size,
                self._cfg.net_stride,
                self._cfg.num_nb)
            tmp_nb_x = lms_pred_nb_x[self._reverse_index1, self._reverse_index2].view(self._cfg.num_lms, self._max_len)
            tmp_nb_y = lms_pred_nb_y[self._reverse_index1, self._reverse_index2].view(self._cfg.num_lms, self._max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred_merge = lms_pred_merge.cpu().numpy()
            pred_export = np.zeros([self._cfg.num_lms, 2])
            for ii in range(self._cfg.num_lms):
                x_pred = lms_pred_merge[ii * 2] * det_width
                y_pred = lms_pred_merge[ii * 2 + 1] * det_height
                pred_export[ii, 0] = x_pred + det_xmin
                pred_export[ii, 1] = y_pred + det_ymin
            landmarks = pred_export

        return landmarks
