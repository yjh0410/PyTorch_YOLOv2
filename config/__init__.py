from .yolov2_config import yolov2_config


def build_model_config(args):
    return yolov2_config[args.version]