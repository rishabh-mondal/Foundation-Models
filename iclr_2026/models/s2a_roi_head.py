# No changes needed. This file is correct.
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .torchvision_utils import Matcher, BalancedPositiveNegativeSampler
from torchvision.ops import box_iou
from .kfiou_loss import KFLoss
from .obb_utils import rotated_nms, decode_deltas_to_rboxes
class S2ARoIHead(nn.Module): # ... (code from previous response)
    def __init__(self, feature_channels: int, num_classes: int, box_roi_pool: nn.Module, score_thresh: float, nms_thresh: float, detections_per_img: int, fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25):
        super(S2ARoIHead, self).__init__(); self.proposal_matcher, self.fg_bg_sampler = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True), BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.box_roi_pool, self.score_thresh, self.nms_thresh, self.detections_per_img, self.regression_clamp_value = box_roi_pool, score_thresh, nms_thresh, detections_per_img, 4.135
        pool_output_size = self.box_roi_pool.output_size[0]; self.align_conv, self.align_regressor = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1), nn.Linear(feature_channels * pool_output_size**2, 5)
        self.refine_head = nn.Sequential(nn.Linear(feature_channels * pool_output_size**2, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU())
        self.cls_predictor, self.bbox_refine_predictor = nn.Linear(1024, num_classes), nn.Linear(1024, 5); self.kfiou_loss, self.cls_loss = KFLoss(reduction='sum', loss_weight=2.0), nn.CrossEntropyLoss()
    def add_gt_proposals(self, p, t): p_gt = []; [p_gt.append(torch.cat((pi,ti["boxes"]),dim=0)) if ti["boxes"].numel()>0 else p_gt.append(pi) for pi,ti in zip(p,t)]; return p_gt
    def select_training_samples(self, p, t):
        p=self.add_gt_proposals(p,t);g_b,g_l=[i["boxes"] for i in t],[i["labels"] for i in t];m_i=[]
        for pi,gi in zip(p,g_b): m_i.append(torch.full((pi.shape[0],),-1,dtype=torch.int64,device=pi.device) if gi.numel()==0 else self.proposal_matcher(box_iou(gi,pi)))
        pos,neg=self.fg_bg_sampler(m_i); s_p,l,r_t=[],[],[]
        for i in range(len(p)):
            pos_i,neg_i=pos[i],neg[i]; inds=torch.cat([pos_i,neg_i],dim=0); s_p.append(p[i][inds]); m_g_i=m_i[i][inds]; l_i=g_l[i][m_g_i.clamp(min=0)]; l_i[len(pos_i):]=0; l.append(l_i)
            p_m_g_i=m_i[i][pos_i]; r=t[i]["rboxes"][p_m_g_i];r_p=torch.zeros((len(inds),5),dtype=r.dtype,device=r.device);r_p[:len(pos_i)]=r; r_t.append(r_p)
        return s_p, l, r_t
    def rbox_to_affine(self, r, o): cx,cy,w,h,a=r.unbind(dim=1);c,s=torch.cos(a),torch.sin(a);sx,sy=w/o[1],h/o[0];th=torch.zeros(r.size(0),2,3,device=r.device);th[:,0,0]=sx*c;th[:,0,1]=-sy*s;th[:,1,0]=sx*s;th[:,1,1]=sy*c;return th
    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training: proposals,labels,regression_targets_rbox = self.select_training_samples(proposals,targets); labels,regression_targets_rbox = torch.cat(labels,dim=0),torch.cat(regression_targets_rbox,dim=0)
        flat_proposals = torch.cat(proposals,dim=0); box_features = self.box_roi_pool(features,proposals,image_shapes); align_features = F.relu(self.align_conv(box_features)).flatten(start_dim=1)
        coarse_deltas = self.align_regressor(align_features).clamp(min=-self.regression_clamp_value,max=self.regression_clamp_value)
        with torch.no_grad(): align_rboxes_src = regression_targets_rbox if self.training else decode_deltas_to_rboxes(flat_proposals,coarse_deltas)
        aligned_box_features = torch.zeros_like(box_features)
        if align_rboxes_src.numel()>0 and align_rboxes_src.size(0)==box_features.size(0): theta = self.rbox_to_affine(align_rboxes_src, box_features.shape[2:]); grid = F.affine_grid(theta,box_features.size(),align_corners=False); aligned_box_features=F.grid_sample(box_features,grid,align_corners=False)
        refined_features = self.refine_head(aligned_box_features.flatten(start_dim=1)); class_logits = self.cls_predictor(refined_features); final_deltas = self.bbox_refine_predictor(refined_features).clamp(min=-self.regression_clamp_value,max=self.regression_clamp_value)
        result,losses = [],{}
        if self.training:
            loss_classifier = self.cls_loss(class_logits,labels); sampled_pos_inds = torch.where(labels>0)[0]
            if sampled_pos_inds.numel()>0:
                loss_reg_coarse = self.kfiou_loss(decode_deltas_to_rboxes(flat_proposals[sampled_pos_inds],coarse_deltas[sampled_pos_inds]),regression_targets_rbox[sampled_pos_inds]); loss_reg_final = self.kfiou_loss(decode_deltas_to_rboxes(align_rboxes_src[sampled_pos_inds],final_deltas[sampled_pos_inds]),regression_targets_rbox[sampled_pos_inds])
            else: loss_reg_coarse,loss_reg_final = torch.tensor(0.,device=coarse_deltas.device),torch.tensor(0.,device=final_deltas.device)
            losses={"loss_classifier":loss_classifier, "loss_reg_coarse":loss_reg_coarse/max(1,sampled_pos_inds.numel()), "loss_reg_final":loss_reg_final/max(1,sampled_pos_inds.numel())}
        else:
            pred_rboxes=decode_deltas_to_rboxes(flat_proposals,final_deltas); pred_scores=F.softmax(class_logits,-1); boxes_per_image=[len(p) for p in proposals]; pred_rboxes_split,pred_scores_split=list(pred_rboxes.split(boxes_per_image,0)),list(pred_scores.split(boxes_per_image,0))
            for i in range(len(proposals)):
                img_boxes,img_scores,img_labels = [],[],[]
                for j in range(1,self.cls_predictor.out_features):
                    inds=torch.where(pred_scores_split[i][:,j]>self.score_thresh)[0];
                    if inds.numel()==0:continue
                    cls_boxes,cls_scores = pred_rboxes_split[i][inds],pred_scores_split[i][inds,j]; keep=rotated_nms(cls_boxes,cls_scores,self.nms_thresh); img_boxes.append(cls_boxes[keep]);img_scores.append(cls_scores[keep]);img_labels.append(torch.full_like(cls_scores[keep],j,dtype=torch.int64))
                if not img_boxes: result.append(dict(rboxes=torch.empty(0,5),scores=torch.empty(0),labels=torch.empty(0))); continue
                img_boxes,img_scores,img_labels = torch.cat(img_boxes),torch.cat(img_scores),torch.cat(img_labels)
                if img_scores.numel()>self.detections_per_img: keep=torch.topk(img_scores,self.detections_per_img,dim=0)[1];img_boxes,img_scores,img_labels=img_boxes[keep],img_scores[keep],img_labels[keep]
                result.append(dict(rboxes=img_boxes,scores=img_scores,labels=img_labels))
        return result,losses