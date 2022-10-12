import os
import torch
import argparse

from data.transform import BaseTransform
from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from utils.misc import load_weight

from config import build_model_config
from models.build import build_yolov2


parser = argparse.ArgumentParser(description='YOLOv2 Detector Evaluation')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')

parser.add_argument('-v', '--version', default='yolov2',
                    help='yolo.')
parser.add_argument('--coco_test', action='store_true', default=False,
                    help='evaluate model on coco-test')
parser.add_argument('--conf_thresh', default=0.001, type=float,
                    help='得分阈值')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS 阈值')
parser.add_argument('--topk', default=1000, type=int,
                    help='topk predicted candidates')
parser.add_argument('--weight', type=str, default=None, 
                    help='Trained state_dict file path to open')

parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')

args = parser.parse_args()



def voc_test(model, device, input_size, val_transform):
    data_root = os.path.join(args.root, 'VOCdevkit')
    evaluator = VOCAPIEvaluator(
        data_root=data_root,
        img_size=input_size,
        device=device,
        transform=val_transform,
        display=True
        )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, device, input_size, val_transform, test=False):
    data_root = os.path.join(args.root, 'COCO')
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=input_size,
            device=device,
            testset=True,
            transform=val_transform
            )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=input_size,
            device=device,
            testset=False,
            transform=val_transform
            )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco':
        print('eval on coco-val ...')
        num_classes = 80
    else:
        print('unknow dataset !! we only support voc, coco !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 构建模型配置文件
    cfg = build_model_config(args)

    # 构建模型
    model = build_yolov2(args, cfg, device, args.input_size, num_classes, trainable=False)

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight)
    model.to(device).eval()
    
    val_transform = BaseTransform(args.input_size)

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, device, args.input_size, val_transform)
        elif args.dataset == 'coco':
            if args.coco_test:
                coco_test(model, device, args.input_size, val_transform, test=True)
            else:
                coco_test(model, device, args.input_size, val_transform, test=False)
