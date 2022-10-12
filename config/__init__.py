from .yolov2_config import yolov2_config


def build_model_config(args):
    if args.version == 'voc':
        dataset = 'voc'
    elif args.version in ['coco', 'coco-val', 'coco-test']:
        dataset = 'coco'
    return yolov2_config[dataset]
    