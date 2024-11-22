import os
from .baseline.config import cfg
from .baseline.model import make_model


def build_reid_model(model_name="resnext101_ibn_a"):
    abs_file = __file__
    print("abs path is %s" % (__file__))
    abs_dir = abs_file[:abs_file.rfind("/")]
    cfg.merge_from_file(os.path.join(abs_dir,'aictest.yml'))
    cfg.INPUT.SIZE_TEST = [384, 384]

    if model_name == "resnext101_ibn_a":
        cfg.MODEL.NAME = 'resnext101_ibn_a'
        model = make_model(cfg, num_class=100)
        model.load_param('post_process/src/reid/reid_model/resnext101_ibn_a_2.pth')
    elif model_name == "resnet101_ibn_a_2":
        cfg.MODEL.NAME = 'resnet101_ibn_a'
        model = make_model(cfg, num_class=100)
        model.load_param('reid/reid_model/resnet101_ibn_a_2.pth')
    elif model_name == "resnet101_ibn_a_3":
        cfg.MODEL.NAME = 'resnet101_ibn_a'
        model = make_model(cfg, num_class=100)
        model.load_param('reid/reid_model/resnet101_ibn_a_3.pth')

    return model,cfg
