# No changes needed. This file is correct.
import torch, cv2, numpy as np
from scipy.spatial import ConvexHull
def decode_deltas_to_rboxes(proposals: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    p_w = proposals[:, 2] - proposals[:, 0]; p_h = proposals[:, 3] - proposals[:, 1]; p_cx = proposals[:, 0] + 0.5 * p_w; p_cy = proposals[:, 1] + 0.5 * p_h
    dx, dy, dw, dh, da = deltas.unbind(dim=1); pred_cx = p_cx + dx * p_w; pred_cy = p_cy + dy * p_h; pred_w = p_w * torch.exp(dw); pred_h = p_h * torch.exp(dh)
    pred_w = pred_w.clamp(min=1e-4); pred_h = pred_h.clamp(min=1e-4); pred_a = da
    return torch.stack((pred_cx, pred_cy, pred_w, pred_h, pred_a), dim=1)
def obb_to_8_points(rboxes: torch.Tensor) -> torch.Tensor: # ... (code from previous response)
    if rboxes.numel() == 0: return torch.empty((0, 8), device=rboxes.device)
    x, y, w, h, a = rboxes.unbind(dim=-1); cosa, sina = torch.cos(a), torch.sin(a); w_half, h_half = w/2, h/2
    dx1, dy1 = -w_half*cosa - h_half*sina,  w_half*sina - h_half*cosa; dx2, dy2 =  w_half*cosa - h_half*sina, -w_half*sina - h_half*cosa
    dx3, dy3 =  w_half*cosa + h_half*sina, -w_half*sina + h_half*cosa; dx4, dy4 = -w_half*cosa + h_half*sina,  w_half*sina + h_half*cosa
    return torch.stack([x+dx1,y+dy1, x+dx2,y+dy2, x+dx3,y+dy3, x+dx4,y+dy4], dim=-1)
def rotated_box_iou(rb1: torch.Tensor, rb2: torch.Tensor) -> torch.Tensor: # ... (code from previous response)
    if rb1.numel() == 0 or rb2.numel() == 0: return torch.zeros((rb1.size(0), rb2.size(0)), device=rb1.device)
    iou_matrix = torch.zeros((rb1.size(0), rb2.size(0))); rb1_cpu, rb2_cpu = rb1.cpu().numpy(), rb2.cpu().numpy()
    for i in range(rb1_cpu.shape[0]):
        for j in range(rb2_cpu.shape[0]):
            rbox1, rbox2 = (rb1_cpu[i,:2],rb1_cpu[i,2:4],rb1_cpu[i,4]*180/np.pi), (rb2_cpu[j,:2],rb2_cpu[j,2:4],rb2_cpu[j,4]*180/np.pi)
            ret, intersection = cv2.rotatedRectangleIntersection(rbox1, rbox2)
            if ret == cv2.INTERSECT_NONE or intersection is None: continue
            try: inter_area = ConvexHull(intersection.squeeze(1)).volume
            except Exception: continue
            area1, area2 = rb1_cpu[i,2]*rb1_cpu[i,3], rb2_cpu[j,2]*rb2_cpu[j,3]; union_area = area1 + area2 - inter_area
            if union_area > 0: iou_matrix[i,j] = inter_area / union_area
    return iou_matrix.to(rb1.device)
def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.4) -> torch.Tensor: # ... (code from previous response)
    if boxes.numel() == 0: return torch.empty(0, dtype=torch.int64)
    order, keep = scores.argsort(descending=True), []
    while order.numel() > 0:
        i = order[0]; keep.append(i.item());
        if order.numel() == 1: break
        iou = rotated_box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0); order = order[torch.where(iou <= iou_threshold)[0] + 1]
    return torch.tensor(keep, dtype=torch.int64)