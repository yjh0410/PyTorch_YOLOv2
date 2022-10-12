from .darknet19 import build_darknet19


def build_backbone(model_name='darknet19', pretrained=False):
    if model_name == 'darknet19':
        backbone, feat_dims = build_darknet19(pretrained)

    return backbone, feat_dims
