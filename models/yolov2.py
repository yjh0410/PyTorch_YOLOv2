import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import Conv, reorg_layer
from backbone import build_backbone

import numpy as np
from .loss import iou_score, compute_loss


class YOLOv2(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 input_size=416,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.001, 
                 nms_thresh=0.6, 
                 topk=100,
                 anchor_size=None):
        super(YOLOv2, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.topk = topk

        # Anchor box config
        self.anchor_size = torch.tensor(anchor_size)  # [KA, 2]
        self.num_anchors = len(anchor_size)
        self.anchor_boxes = self.create_grid(input_size)

        # 主干网络：resnet50
        self.backbone, feat_dims = build_backbone(cfg['backbone'], cfg['pretrained'])
        
        # 检测头
        self.convsets_1 = nn.Sequential(
            Conv(feat_dims[-1], cfg['head_dim'], k=3, p=1),
            Conv(cfg['head_dim'], cfg['head_dim'], k=3, p=1)
        )

        # 融合高分辨率的特征信息
        self.route_layer = Conv(feat_dims[-2], cfg['reorg_dim'], k=1)
        self.reorg = reorg_layer(stride=2)

        # 检测头
        self.convsets_2 = Conv(cfg['head_dim']+cfg['reorg_dim']*4, cfg['head_dim'], k=3, p=1)
        
        # 预测曾
        self.pred = nn.Conv2d(cfg['head_dim'], self.num_anchors*(1 + 4 + self.num_classes), 1)


        if self.trainable:
            self.init_bias()


    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.pred.bias[..., 1*self.num_anchors:(1+self.num_classes)*self.num_anchors], bias_value)


    def create_grid(self, input_size):
        w, h = input_size, input_size
        # 生成G矩阵
        fmp_w, fmp_h = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2]
        grid_xy = grid_xy[:, None, :].repeat(1, self.num_anchors, 1)

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([grid_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes        


    def set_grid(self, input_size):
        self.input_size = input_size
        self.anchor_boxes = self.create_grid(input_size)


    def decode_boxes(self, anchors, txtytwth_pred):
        """将txtytwth预测换算成边界框的左上角点坐标和右下角点坐标 \n
            Input: \n
                txtytwth_pred : [B, H*W*KA, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*KA, 4] \n
        """
        # 获得边界框的中心点坐标和宽高
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[..., :2]) + anchors[..., :2]
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[..., 2:]) * anchors[..., 2:]

        # [B, H*W*KA, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1) * self.stride

        # 将中心点坐标和宽高换算成边界框的左上角点坐标和右下角点坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[..., :2] = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred[..., 2:] = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        
        return x1y1x2y2_pred


    def nms(self, bboxes, scores):
        """"Pure Python NMS baseline."""
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, conf_pred, cls_pred, reg_pred):
        """
        Input:
            conf_pred: (Tensor) [H*W*KA, 1]
            cls_pred:  (Tensor) [H*W*KA, C]
            reg_pred:  (Tensor) [H*W*KA, 4]
        """
        anchors = self.anchor_boxes

        # (H x W x KA x C,)
        scores = (torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)).flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, reg_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = scores.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        labels = topk_idxs % self.num_classes

        reg_pred = reg_pred[anchor_idxs]
        anchors = anchors[anchor_idxs]

        # 解算边界框, 并归一化边界框: [H*W*KA, 4]
        bboxes = self.decode_boxes(anchors, reg_pred)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 归一化边界框
        bboxes = bboxes / self.input_size
        bboxes = np.clip(bboxes, 0., 1.)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # backbone主干网络
        feats = self.backbone(x)
        c4, c5 = feats['c4'], feats['c5']

        # 处理c5特征
        p5 = self.convsets_1(c5)

        # 融合c4特征
        p4 = self.reorg(self.route_layer(c4))
        p5 = torch.cat([p4, p5], dim=1)

        # 处理p5特征
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)

        B, abC, H, W = prediction.size()
        KA = self.num_anchors
        NC = self.num_classes

        # [B, KA * C, H, W] -> [B, H, W, KA * C] -> [B, H*W, KA*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W, KA*C] -> [B, H*W, KA] -> [B, H*W*KA, 1]
        conf_pred = prediction[..., :KA].contiguous().view(B, -1, 1)
        # [B, H*W, KA*C] -> [B, H*W, KA*NC] -> [B, H*W*KA, NC]
        cls_pred = prediction[..., 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)
        # [B, H*W, KA*C] -> [B, H*W, KA*4] -> [B, H*W, KA, 4]
        txtytwth_pred = prediction[..., (1+NC)*KA:].contiguous().view(B, -1, 4)

        # 测试时，笔者默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0]            #[H*W*KA, 1]
        cls_pred = cls_pred[0]              #[H*W*KA, NC]
        txtytwth_pred = txtytwth_pred[0]    #[H*W*KA, 4]

        # 后处理
        bboxes, scores, labels = self.postprocess(conf_pred, cls_pred, txtytwth_pred)

        return bboxes, scores, labels


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            feats = self.backbone(x)
            c4, c5 = feats['c4'], feats['c5']

            # 处理c5特征
            p5 = self.convsets_1(c5)

            # 融合c4特征
            p4 = self.reorg(self.route_layer(c4))
            p5 = torch.cat([p4, p5], dim=1)

            # 处理p5特征
            p5 = self.convsets_2(p5)

            # 预测
            prediction = self.pred(p5)

            B, abC, H, W = prediction.size()
            KA = self.num_anchors
            NC = self.num_classes

            # [B, KA * C, H, W] -> [B, H, W, KA * C] -> [B, H*W, KA*C]
            prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
            # [B, H*W, KA*C] -> [B, H*W, KA] -> [B, H*W*KA, 1]
            conf_pred = prediction[..., :KA].contiguous().view(B, -1, 1)
            # [B, H*W, KA*C] -> [B, H*W, KA*NC] -> [B, H*W*KA, NC]
            cls_pred = prediction[..., 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)
            # [B, H*W, KA*C] -> [B, H*W, KA*4] -> [B, H*W*KA, 4]
            txtytwth_pred = prediction[..., (1+NC)*KA:].contiguous().view(B, -1, 4)  

            # 解算边界框
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = targets[:, :, 7:].view(-1, 4)

            # 计算预测框和真实框之间的IoU
            iou_pred = iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

            # 将IoU作为置信度的学习目标
            with torch.no_grad():
                gt_conf = iou_pred.clone()
            
            # 将IoU作为置信度的学习目标 
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            targets = torch.cat([gt_conf, targets[:, :, :7]], dim=2)

            # 计算损失
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(
                pred_conf=conf_pred, 
                pred_cls=cls_pred,
                pred_txtytwth=txtytwth_pred,
                targets=targets,
                )

            return conf_loss, cls_loss, bbox_loss, total_loss
