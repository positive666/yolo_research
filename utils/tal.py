#From Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import check_version
from utils.metrics import bbox_iou

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """  
    """该函数的作用是通过码头坐标和真实框的位置信息，在所有anchor中选择位于真实框内部或者与其IoU大于阈值的anchor点，并返回一个(b, n_boxes, h*w)的张量表示所选择的anchor点。
    下面对该函数的每句话代码进行注释和讲解："""
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape  # 每个GT有多少个Anchor
    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            """根据左上角和右下角的坐标值计算出目标box与所有anchor的平移和缩放量，拼接后+reshape为(n_boxes, n_anchors, -1) 在
            取其偏移量维度的数值，取最小值,anchor标记为 positive,判断其中每一行的值是否大于0,否则为负样本"""
            bbox_deltas[b] = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),    ## 计算anchor和GT角点偏移和缩放，广播对齐
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else: ##同理计算方式不同
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1) # 形状变为(b, n_boxes, n_anchors, 4)
        #return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype), 计算最小值 返回一个大小为(b, n_boxes, h*w)的张量
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes): #解决一个anchor和多个目标框匹配的问题
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)，表示对应的anchor是否与目标框匹配
        overlaps (Tensor): shape(b, n_max_boxes, h*w)，表示对应的anchor和所有目标框的重叠面积。
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w) 
    fg_mask = mask_pos.sum(-2)  # 沿着第2维度求和，即可得到一个大小为(b, h * w)的张量 fg_mask，每个元素表示anchor匹配目标框的数
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes:一个anchor分配了多个gt_box
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w):值为1的位置表示该anchor所在格子分配多个gt_box
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)，取与对应anchor重叠面积最大的目标框的下标，得到一个大小为 (batch_size, h * w) 的张量 max_overlaps_idx索引
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)：max_overlaps_idx 转换 one-hot,某个anchor上的最大IOU。
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w): 其值表示每个anchor应分配给哪个gt_box
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w): 逐元素比较,其值为True的位置表示需选择IoU最高的gt_box
        fg_mask = mask_pos.sum(-2)     # 每个元素表示对应的anchor匹配的哪个gt_box:正样本
    # find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w): 取索引
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes     
        self.bg_idx = num_classes
        self.alpha = alpha                 # match  weigth
        self.beta = beta                   # IOU    weight 
        self.eps = eps
        self.roll_out_thr = roll_out_thr   # roll threshold

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)   
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        self.roll_out = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))
        # 获取正样本掩码，、匹配度、重叠度
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)
        # get target IOU match：解决一个anchor和多个GT框匹配问题
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target：分配标签
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize：归一化
        align_metric *= mask_pos #mask_pos每个anchor box是否与gt_box相交，如果相交，则为1，否则为0, 等价没有分配gt_box的anchorbox的对齐度量设置为0
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj：gt box和所有anchor box对应的最大度量值，
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj：gt box和所有anchor box之间的最大IoU值
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1) #每个anchor box的归一化因子：归一化目标分数target_scores
        target_scores = target_scores * norm_align_metric  

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (b, max_num_obj, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        # get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """ 两个计算方式，目前来看ROLL OUT逻辑上更快"""
        if self.roll_out:     
            align_metric = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0 = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                """ gt_labes info --->improve (使用 roll_out 策略时，只计算那些被标签所覆盖的边框与 GT 之间的 CIoU,减少了计算量
                而对于那些不被 ground truth 标签所覆盖的边框，将被舍弃，避免了计算冗余和过多内存消耗)   """
                # form gt_label    
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()   
                # get the scores of each grid for each gt cls
                bbox_scores = pd_scores[ind_0, :, ind_2]  # b, max_num_obj, h*w  ,表示第某batch的某GT对应的某个网格的得分
                # Calculate  CIoU per grid and per GT，(only marked boxes)
                overlaps[b] = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False,
                                       CIoU=True).squeeze(2).clamp(0)
                # align match metric (only gt && d)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)  
        # Boardcast加快计算效率 
        else:
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
            ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
            # get the scores of each grid for each gt cls
            bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w
            # 计算每个网格和每个GT的IoU（或CIoU）/# Calculate  CIoU per grid and per GT
            overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False,
                                CIoU=True).squeeze(3).clamp(0)
            # 对齐度量值计算/match metric
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            """ 如果 topk_mask 为 None，则 topk_metrics 和 self.eps 的逻辑值生成一个大小为 (b, max_num_obj, topk) 的逻辑掩码 """
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)  """  
        topk_idxs[~topk_mask] = 0
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        """  两种计算方式 ，roll out降低开销，no roll out 计算更快"""
        if self.roll_out:    
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
            根据分配给每个anchor box的gt_box,得到其对应的目标类别、目标框和目标分数
        """

        # assigned target labels, (b, 1)，batch索引
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)：target_gt_idx加上该batch索引和偏移量
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)：从gt_labels中选择对应labels

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)：gt_bboxes中选择对应的boxes
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp_(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80) 
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80):第i个元素表示第i个anchor box是否是前景
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0) # 比较，选取前景的anchorbox对应的one-hot向量，其余置为0

        return target_labels, target_bboxes, target_scores


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1): 
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)     
    x1y1 = anchor_points - lt                  #计算坐标
    x2y2 = anchor_points + rb
    if xywh:                                 ##bOX 格式转换
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
