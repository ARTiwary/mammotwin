"""
Phase 7: Faster R-CNN lesion detector (single class: "lesion" vs background),
built on torchvision's pretrained detection backbone per the project plan's
suggested models (YOLO-style / Faster R-CNN / RetinaNet / MONAI detection).
Faster R-CNN was chosen because torchvision ships a ready, well-tested
implementation, minimizing custom detection-head code.
"""

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(config: dict):
    pretrained = config["model"].get("pretrained", True)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    # torchvision's builder downloads an ImageNet-pretrained ResNet50 BACKBONE
    # by default even when `weights=None` (that only controls the detection
    # head). Must explicitly pass weights_backbone=None too to fully disable
    # any pretrained-weights download.
    weights_backbone = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=weights_backbone)

    num_classes = 2  # 0 = background, 1 = lesion
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model