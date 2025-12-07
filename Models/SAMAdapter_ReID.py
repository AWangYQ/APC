import logging
import torch
import math
from torch.nn import functional as F
from Models.SAM.segment_anything.modeling.image_encoder import ImageEncoderViT
from torch import nn
from Models.backbone import clip
from Models.backbone.TextEncoder import TextEncoder
import random
from timm.models.layers import trunc_normal_
import ipdb


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(
        posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

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
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.IN = nn.LayerNorm(c_in)

    def forward(self, x):
        if len(x.size()) == 3:
            return self.fc(self.IN(x))
        else:
            b, h, w, c = x.size()
            out = self.fc(self.IN(x.view(b, h*w, c)))
            return out.view(b, h, w, c)

class FusionDecoder(nn.Module):

    def __init__(self):
        super(FusionDecoder, self).__init__()
        self.DecoderSelfAttention = nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1)
        self.Decoderlinear1 = nn.Linear(768, 2048)
        self.Decoderdropout = nn.Dropout(0.1)
        self.Decoderlinear2 = nn.Linear(2048, 768)

        self.Decodernorm1 = nn.LayerNorm(768)
        self.Decodernorm2 = nn.LayerNorm(768)
        self.Decoderdropout1 = nn.Dropout(0.1)
        self.Decoderdropout2 = nn.Dropout(0.1)

        self.activation = F.relu


    def forward(self, x):
        x = self.Decoderdropout1(self.DecoderSelfAttention(x, x, x)[0])
        x = self.Decodernorm1(x)
        x = self.Decoderlinear2(self.Decoderdropout(self.activation(self.Decoderlinear1(x))))  # 这里没有短接，不知道用了短接效果会怎么样
        x = self.Decodernorm2(self.Decoderdropout2(x))
        return x


class SAMAdapterReID(nn.ModuleList):

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

    def __init__(self, cfg, num_classes, camera_num, view_num, __prompts):
        super().__init__()

        self.logger = logging.getLogger("SAMAdapterReID.Module")
        self.in_planes = 768
        self.num_classes = num_classes       # 分类数
        self.camera_num = camera_num         # 相机数
        self.view_num = view_num             # 视角数
        self.sie_coe = cfg.MODEL.SIE_COE     # SIE 信息的权重
        self.model_name = 'ViT-B-16'         # 设置模型名字
        self.prompts = __prompts

        # 定义分类器
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.clip_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.clip_classifier.apply(weights_init_classifier)

        self.text_classifier = nn.Linear(768, self.num_classes, bias=False)
        self.text_classifier.apply(weights_init_classifier)

        # 定义BN层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.clip_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.clip_bottleneck.bias.requires_grad_(False)
        self.clip_bottleneck.apply(weights_init_kaiming)

        self.text_bottleneck = nn.BatchNorm1d(512)
        self.text_bottleneck.bias.requires_grad_(False)
        self.text_bottleneck.apply(weights_init_kaiming)

        # 定义 adapter 层
        self.clip_adapter = nn.ModuleList([Adapter(self.in_planes, 12) for _ in range(12)])

        # 定义clip模型
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-cfg.MODEL.PATCH_SIZE[0])//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-cfg.MODEL.PATCH_SIZE[1])//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = self.load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.clip_image_encoder = clip_model.visual  # 这里是clip版本的VIT，使用的时候，注意他们用的一些tricks
        if self.camera_num > 1 and self.view_num > 1:
            self.sie_embed = nn.Parameter(torch.zeros(self.camera_num * self.view_num, 1, self.in_planes))
            trunc_normal_(self.sie_embed, std=.02)
            self.logger.info(
                'camera number is : {} and viewpoint number is : {}'.format(self.camera_num, self.view_num))
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

        # fusion 结构
        self.FusionDecoder1 = FusionDecoder()
        # self.FusionDecoder2 = FusionDecoder()
        # self.ClipDecoderIN = nn.LayerNorm(768)

        self.activation = F.relu
        scale = 768 ** -0.5
        self.text_prompt = nn.Parameter(torch.randn(77, 768))

        cross_attention = nn.TransformerDecoderLayer(d_model=768, nhead=12, batch_first=True)
        self.cross_prompt = nn.TransformerDecoder(decoder_layer=cross_attention, num_layers=2)
        # self.prompt_proj = nn.Linear(512, 768)
        # self.text_prompt = nn.Parameter(scale * torch.randn(10, 768))
        # self.logit_scale = clip_model.logit_scale

        # self.task_prompt = nn.Parameter(scale * torch.randn(768))
        # self.MSE_LOSS = nn.MSELoss()


    def load_param(self, model_path):

        state_dict = torch.load(model_path, map_location='cpu')

        # 将pretrained_dict里不属于model_dict的键剔除掉  慎重使用！！！
        # param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}
        param_dict = {k:v for k, v in state_dict.items()}
        # ipdb.set_trace()

        # if 'model' in param_dict:
        #     param_dict = param_dict['model']
        # if 'state_dict' in param_dict:
        #     param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,self.state_dict()[k].shape))



    def forward(self, x=None, cam_label=None, view_label=None,backbone_feat=False):

        # if cam_label != None and view_label != None:
        #     cv_embed = self.sie_coe * self.sie_embed[cam_label * self.view_num + view_label]
        # elif cam_label != None:
        #     cv_embed = self.sie_coe * self.sie_embed[cam_label]
        # elif view_label != None:
        #     cv_embed = self.sie_coe * self.sie_embed[view_label]
        # else:
        cv_embed = None

        clip_11, clip_12, xproj = self.clip_image_encoder(x, cv_embed, self.clip_adapter)

        aggregated_features = self.FusionDecoder1(clip_12)

        text_features = self.text_prompt.repeat(x.size(0), 1, 1)
        text_features = self.cross_prompt(text_features, clip_12).mean(1)

        cls = aggregated_features[:, 0].view(x.size(0), -1)
        bn_cls = self.bottleneck(cls)

        clip_cls = clip_12[:, 0].view(x.size(0), -1)
        bn_clip_cls = self.clip_bottleneck(clip_cls)

        bn_text_features = self.text_bottleneck(text_features)

        cls_score = []
        feats = []
        if self.training:
            feats.append(bn_cls)
            feats.append(bn_clip_cls)
            feats.append(bn_text_features)
            cls_score.append(self.classifier(bn_cls))
            cls_score.append(self.clip_classifier(bn_clip_cls))
            cls_score.append(self.text_classifier(bn_text_features))
            return cls_score, feats
        else:
            return torch.cat([cls, clip_cls, text_features], 1)