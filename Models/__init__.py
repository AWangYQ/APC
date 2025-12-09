from Models.APC import APC

__models = {
    'APC': APC
}

def make_model(cfg, num_classes, camera_num, view_num):
    return __models[cfg.MODEL.NAME](cfg, num_classes, camera_num, view_num, adapter=False)
