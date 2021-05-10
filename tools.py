import numpy as np
from data import *
import torch.nn as nn
import torch.nn.functional as F

# We use ignore thresh to decide which anchor box can be kept.
ignore_thresh = IGNORE_THRESH


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        # 被忽略的先验框的mask都是-1，不参与loss计算
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


    def compute_iou(anchor_boxes, gt_box):
        """计算先验框和真实框之间的IoU
        Input: \n
            anchor_boxes: [N, 4] \n
                gt_box: [1, 4] \n
        Output: \n
                    iou : [N, 1] \n
        """

        # anchor box :
        ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
        # 计算先验框的左上角点坐标和右下角点坐标
        ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
        ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
        ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
        ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
        w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
        
        # gt_box : 
        # 我们将真实框扩展成[N, 4], 便于计算IoU. 
        gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

        gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
        # 计算真实框的左上角点坐标和右下角点坐标
        gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
        gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
        gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
        gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
        w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

        # 计算IoU
        S_gt = w_gt * h_gt
        S_ab = w_ab * h_ab
        I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
        I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
        S_I = I_h * I_w
        U = S_gt + S_ab - S_I + 1e-20
        IoU = S_I / U
        
        return IoU


def set_anchors(anchor_size):
    """将输入进来的只包含wh的先验框尺寸转换成[N, 4]的ndarray类型，
       包含先验框的中心点坐标和宽高wh，中心点坐标设为0. \n
    Input: \n
        anchor_size: list -> [[h_1, w_1],  \n
                              [h_2, w_2],  \n
                               ...,  \n
                              [h_n, w_n]]. \n
    Output: \n
        anchor_boxes: ndarray -> [[0, 0, anchor_w, anchor_h], \n
                                  [0, 0, anchor_w, anchor_h], \n
                                  ... \n
                                  [0, 0, anchor_w, anchor_h]]. \n
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def generate_txtytwth(gt_label, w, h, s, anchor_size):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算真实边界框的中心点和宽高
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-4 or box_h < 1e-4:
        # print('not a valid data !!!')
        return False    

    # 将真是边界框的尺寸映射到网格的尺度上去
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s
    
    # 计算中心点所落在的网格的坐标
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)

    # 获得先验框的中心点坐标和宽高，
    # 这里，我们设置所有的先验框的中心点坐标为0
    anchor_boxes = set_anchors(anchor_size)
    gt_box = np.array([[0, 0, box_ws, box_hs]])

    # 计算先验框和真实框之间的IoU
    iou = compute_iou(anchor_boxes, gt_box)

    # 只保留大于ignore_thresh的先验框去做正样本匹配,
    iou_mask = (iou > ignore_thresh)

    result = []
    if iou_mask.sum() == 0:
        # 如果所有的先验框算出的IoU都小于阈值，那么就将IoU最大的那个先验框分配给正样本.
        # 其他的先验框统统视为负样本
        index = np.argmax(iou)
        p_w, p_h = anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w)
        th = np.log(box_hs / p_h)
        weight = 2.0 - (box_w / w) * (box_h / h)
        
        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
        
        return result
    
    else:
        # 有至少一个先验框的IoU超过了阈值.
        # 但我们只保留超过阈值的那些先验框中IoU最大的，其他的先验框忽略掉，不参与loss计算。
        # 而小于阈值的先验框统统视为负样本。
        best_index = np.argmax(iou)
        for index, iou_m in enumerate(iou_mask):
            if iou_m:
                if index == best_index:
                    p_w, p_h = anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w)
                    th = np.log(box_hs / p_h)
                    weight = 2.0 - (box_w / w) * (box_h / h)
                    
                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    # 对于被忽略的先验框，我们将其权重weight设置为-1
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

        return result 


def gt_creator(input_size, stride, label_lists, anchor_size):
    # 必要的参数
    batch_size = len(label_lists)
    s = stride
    w = input_size
    h = input_size
    ws = w // s
    hs = h // s
    anchor_number = len(anchor_size)
    gt_tensor = np.zeros([batch_size, hs, ws, anchor_number, 1+1+4+1+4])

    # 制作正样本
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            results = generate_txtytwth(gt_label, w, h, s, anchor_size)
            if results:
                for result in results:
                    index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0.:
                        if grid_y < gt_tensor.shape[1] and grid_x < gt_tensor.shape[2]:
                            gt_tensor[batch_index, grid_y, grid_x, index, 0] = 1.0
                            gt_tensor[batch_index, grid_y, grid_x, index, 1] = gt_class
                            gt_tensor[batch_index, grid_y, grid_x, index, 2:6] = np.array([tx, ty, tw, th])
                            gt_tensor[batch_index, grid_y, grid_x, index, 6] = weight
                            gt_tensor[batch_index, grid_y, grid_x, index, 7:] = np.array([xmin, ymin, xmax, ymax])
                    else:
                        # 对于那些被忽略的先验框，其gt_obj参数为-1，weight权重也是-1
                        gt_tensor[batch_index, grid_y, grid_x, index, 0] = -1.0
                        gt_tensor[batch_index, grid_y, grid_x, index, 6] = -1.0

    gt_tensor = gt_tensor.reshape(batch_size, hs * ws * anchor_number, 1+1+4+1+4)

    return gt_tensor


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i)


def loss(pred_conf, pred_cls, pred_txtytwth, pred_iou, label, num_classes):
    # 损失函数
    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')
    iou_loss_function = nn.SmoothL1Loss(reduction='none')

    # 预测
    pred_conf = pred_conf[:, :, 0]
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
    pred_iou = pred_iou[:, :, 0]

    # 标签  
    gt_conf = label[:, :, 0].float()
    gt_obj = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txty = label[:, :, 3:5].float()
    gt_twth = label[:, :, 5:7].float()
    gt_box_scale_weight = label[:, :, 7]
    gt_iou = (gt_box_scale_weight > 0.).float()
    gt_mask = (gt_box_scale_weight > 0.).float()

    batch_size = pred_conf.size(0)
    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    
    # 类别损失
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask) / batch_size
    
    # 边界框的位置损失
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
    bbox_loss = txty_loss + twth_loss

    # iou 损失
    iou_loss = torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

    return conf_loss, cls_loss, bbox_loss, iou_loss


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)