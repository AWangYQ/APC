from Models.CnnBase import ResNet50
from Models.VitBase import Vit
from Models.ClipBase import ClipAdapter
from Models.SAMAdapter_ReID import SAMAdapterReID

__models = {
    'cnn_base': ResNet50,
    'vit_base': Vit,
    'SAMAdapter_ReID': SAMAdapterReID,
    'clip_adapter': ClipAdapter
}

def make_model(cfg, num_classes, camera_num, view_num):
    return __models[cfg.MODEL.NAME](cfg, num_classes, camera_num, view_num, adapter=False)