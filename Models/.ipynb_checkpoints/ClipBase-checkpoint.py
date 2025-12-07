import ipdb
import torch
from torch import nn
# import clip
from Models.backbone import clip
import logging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=12):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.IN = nn.LayerNorm(c_in)

    def forward(self, x):
        return self.fc(self.IN(x))

class ClipAdapter(nn.Module):

    def __init__(self, cfg, num_classes, camera_num, view_num, adapter):
        super(ClipAdapter, self).__init__()

        self.logger = logging.getLogger("ReIDAdapter.Model")
        self.in_planes = 768                 # 特征向量的维度
        self.in_planes_proj = 512            # 模态对齐特征向量的维度
        self.num_classes = num_classes       # 分类数
        self.camera_num = camera_num         # 相机数
        self.view_num = view_num             # 视角数
        self.sie_coe = cfg.MODEL.SIE_COE     # SIE 信息的权重
        self.model_name = 'ViT-B-16'         # 设置模型名字

        # 定义分类器
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        # 定义BN层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.temperature = 0.007

        # 定义clip模型
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-cfg.MODEL.PATCH_SIZE[0])//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-cfg.MODEL.PATCH_SIZE[1])//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = self.load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        # 定义 adapter 层
        if adapter:
            self.adapter = nn.ModuleList([Adapter(self.in_planes, 4) for _ in range(12)])
        else:
            self.adapter = None

        self.image_encoder = clip_model.visual     # 这里是clip版本的VIT，使用的时候，注意他们用的一些tricks

        if self.camera_num > 1 and self.view_num > 1:
            self.sie_embed = nn.Parameter(torch.zeros(self.camera_num * self.view_num, 1, self.in_planes))
            trunc_normal_(self.sie_embed, std=.02)
            self.logger.info('camera number is : {} and viewpoint number is : {}'.format(self.camera_num, self.view_num))
            self.logger.info('using SIE_Lambda is : {}'.format(self.sie_coe))
        elif self.camera_num > 1:
            self.sie_embed = nn.Parameter(torch.zeros(self.camera_num, self.in_planes))
            trunc_normal_(self.sie_embed, std=.02)
            self.logger.info('camera number is : {}'.format(self.camera_num))
            self.logger.info('using SIE_Lambda is : {}'.format(self.sie_coe))
        elif self.view_num > 1:
            self.sie_embed = nn.Parameter(torch.zeros(self.view_num, self.in_planes))
            trunc_normal_(self.sie_embed, std=.02)
            self.logger.info('viewpoint number is : {}'.format(self.view_num))
            self.logger.info('using SIE_Lambda is : {}'.format(self.sie_coe))


    def load_clip_to_cpu(self, backbone_name, h_resolution, w_resolution, vision_stride_size):
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

        return model

    def forward(self, x=None, cam_label=None, view_label=None, classifer=None):

        # b, c, h, w = x.size()
        if cam_label != None and view_label != None:
            cv_embed = self.sie_coe * self.sie_embed[cam_label * self.view_num + view_label]
        elif cam_label != None:
            cv_embed = self.sie_coe * self.sie_embed[cam_label]
        elif view_label != None:
            cv_embed = self.sie_coe * self.sie_embed[view_label]
        else:
            cv_embed = None

        x_11, x_12, xproj = self.image_encoder(x, cv_embed, self.adapter)
        cls = x_12[:, 0]

        if self.training:
            feat = cls / cls.norm(dim=-1, keepdim=True)
            classifer = classifer / classifer.norm(dim=-1, keepdim=True)
            cls_score = (feat @ classifer.t())/ self.temperature
            bn_cls = self.bottleneck(cls)
            cls_score1 = self.classifier(bn_cls)
            return [cls_score, cls_score1], [cls]
        else:
            # feat = cls / cls.norm(dim=-1, keepdim=True)
            return cls

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        self.logger.info('Loading pretrained model from {}'.format(trained_path))