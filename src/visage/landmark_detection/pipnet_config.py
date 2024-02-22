from dataclasses import dataclass


@dataclass
class PIPNetConfig:
    backbone: str
    pretrained: bool
    num_nb: int
    num_lms: int  # Number of landmarks
    input_size: int
    net_stride: int
    checkpoint_path: str  # Path inside $HOME/.cache/torch/face-parser/PIPNet where checkpoint is located
    meanface_path: str  # Path inside face_parser.PIPNet.meanfaces where meanface specifications are stored


PIPNet_WFLW_r18_config = PIPNetConfig(
    backbone='resnet18',
    pretrained=True,
    num_nb=10,
    num_lms=98,
    input_size=256,
    net_stride=32,
    checkpoint_path='WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth',
    meanface_path='WFLW/meanface.txt'
)
