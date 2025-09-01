# No changes needed. This file is correct.
import torch, torch.nn as nn
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from terratorch.registry import BACKBONE_REGISTRY
from .s2a_roi_head import S2ARoIHead
class TerraMindBackboneWrapper(nn.Module): # ... (code from previous response)
    def __init__(self): super().__init__(); self.backbone = BACKBONE_REGISTRY.build("terramind_v1_base", modalities=["S2L2A"], pretrained=True, bands={"S2L2A": ["B4", "B3", "B2"]}); self.out_channels = 768; self.return_layers_indices = [3, 5, 8, 11]
    def forward(self, x: torch.Tensor) -> OrderedDict:
        features_list = self.backbone({"S2L2A": x}); out = OrderedDict()
        for i, layer_idx in enumerate(self.return_layers_indices): tokens = features_list[layer_idx]; h_w = int(tokens.shape[1]**0.5); out[str(i)] = tokens.permute(0, 2, 1).reshape(x.shape[0], self.out_channels, h_w, h_w)
        return out
class OrientedRCNN(nn.Module): # ... (code from previous response)
    def __init__(self, backbone, rpn, roi_heads, transform): super().__init__(); self.transform, self.backbone, self.rpn, self.roi_heads = transform, backbone, rpn, roi_heads
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if self.training and targets is None: raise ValueError("In training mode, targets should be passed")
        original_image_sizes: List[Tuple[int, int]] = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets); features = self.backbone(images.tensors); proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        if not self.training: [det.update({'boxes': det.pop('rboxes')}) for det in detections]
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes); losses = {}; losses.update(detector_losses); losses.update(proposal_losses)
        return losses if self.training else detections
def create_oriented_rcnn_model(num_classes: int, input_size: int = 224): # ... (code from previous response)
    backbone = TerraMindBackboneWrapper()
    anchor_generator = AnchorGenerator(sizes=((32,64,128,256),)*len(backbone.return_layers_indices), aspect_ratios=((0.5,1.0,2.0),)*len(backbone.return_layers_indices))
    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0]); rpn = RegionProposalNetwork(anchor_generator,rpn_head,fg_iou_thresh=0.7,bg_iou_thresh=0.3,batch_size_per_image=256,positive_fraction=0.5,pre_nms_top_n=dict(training=2000,testing=1000),post_nms_top_n=dict(training=2000,testing=1000),nms_thresh=0.7)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1','2','3'], output_size=7, sampling_ratio=2)
    s2a_head = S2ARoIHead(feature_channels=backbone.out_channels,num_classes=num_classes,box_roi_pool=roi_pooler,score_thresh=0.05,nms_thresh=0.4,detections_per_img=100)
    transform = GeneralizedRCNNTransform(min_size=input_size,max_size=input_size,image_mean=[0.485,0.456,0.406],image_std=[0.229,0.224,0.225],fixed_size=(input_size,input_size))
    return OrientedRCNN(backbone, rpn, s2a_head, transform)