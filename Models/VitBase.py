import logging
import torch
import torch.nn as nn
import copy

from Models.backbone.vit_pytorch import vit_base_patch16_224_TransReID as ViT_B

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

class Vit(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super(Vit, self).__init__()
        self.logger = logging.getLogger("ReIDAdapter.Model")
        model_path = cfg.MODEL.PRETRAIN_PATH
        self.in_planes = 768

        self.logger.info('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = ViT_B(self.logger, img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        self.base.load_param(model_path)
        self.logger.info('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def video_prepare(self, x):
        b, t, c, h, w = x.shape
        return x.view(b*t, c, h, w)

    def video_postprocessing(self, x, t):
        c = x.size(1)
        return x.view(-1, t, c).mean(1)

    def forward(self, x, cam_label= None, view_label=None, type='image'):

        if type == 'video':
            x,t = self.video_prepare(x)

        feature = self.base(x, cam_label=cam_label, view_label=view_label)

        if type == 'video':
            feature = self.video_postprocessing(feature)

        feat = self.bottleneck(feature)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feature  # global feature for triplet loss
        else:
            return feature

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        self.logger.info('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        self.logger.info('Loading pretrained model for finetuning from {}'.format(model_path))
