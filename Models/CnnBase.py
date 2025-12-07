import torch.nn as nn
from torchvision import models
import torch

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

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        return self.avgpooling(x).view(x.size(0), x.size(1))

class ResNet50(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super(ResNet50, self).__init__()

        self.dim = 2048
        self.num_classes = num_classes
        # 定义加载了预训练模型的resnet50
        resnet = models.resnet50(pretrained=True)

        # 将预训练过后的renset50作为特征提取器，并且将最后一层的卷积步长设置为 1
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone[-1][0].conv2.stride = (1,1)
        self.backbone[-1][0].downsample[0].stride = (1,1)

        # Globally Average Pooling
        self.GAP = Pooling()

        # 定义分类器
        self.bottleneck = nn.BatchNorm1d(self.dim)
        self.classifier = nn.Linear(self.dim,self.num_classes,bias=False)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def video_prepare(self, x):
        b, t, c, h, w = x.shape
        return x.view(b*t, c, h, w)

    def video_postprocessing(self, x, t):
        c = x.size(1)
        return x.view(-1, t, c).mean(1)

    def feature_extractor(self,x):
        return self.backbone(x)

    def forward(self, x, cam_label, view_label, type):

        if type == 'video':
            x,t = self.video_prepare(x)

        featmap = self.feature_extractor(x)
        feature = self.GAP(featmap)

        if type == 'video':
            feature = self.video_postprocessing(feature)

        if self.training:
            bn_feat = self.bottleneck(feature)
            cls = self.classifier(bn_feat)
            return cls, feature
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