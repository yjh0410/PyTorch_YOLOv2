from .yolov2 import YOLOv2


def build_yolov2(args, cfg, device, input_size, num_classes=20, trainable=False):
    anchor_size = cfg['anchor_size'][args.dataset]
    
    model = YOLOv2(
        cfg=cfg,
        device=device,
        input_size=input_size,
        num_classes=num_classes,
        trainable=trainable,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.topk,
        anchor_size=anchor_size
        )

    return model
