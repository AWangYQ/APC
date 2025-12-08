import ipdb
import torch
from torch import nn
from collections import OrderedDict
from Models.backbone.model import QuickGELU 
# import clip
from Models.backbone import clip
import logging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F 
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
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts=None, eos_token_pos=None, text=None):
        if text is not None:
            x = self.token_embedding(text).type(self.dtype)
        elif prompts is not None:
            x = prompts
        else:
            raise " You need inputs ! "
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if eos_token_pos is not None:
            x = x[torch.arange(x.shape[0]), eos_token_pos] @ self.text_projection
        elif text is not None:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 
        else:
            raise " You need check it out! "
        return x

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

class APC(nn.Module):

    def __init__(self, cfg, num_classes, camera_num, view_num, adapter):
        super(APC, self).__init__()

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
        self.h_classifier = nn.Linear(512, self.num_classes, bias=False)
        self.h_classifier.apply(weights_init_classifier)
        # 定义BN层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(512)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)
 
        self.temperature = 0.007

        #self.attribute_selfformer = nn.ModuleList([nn.MultiheadAttention(768, 8, batch_first=True) for _ in range(3)])
        #self.attribute_selfnorm = nn.ModuleList([nn.LayerNorm(768) for _ in range(3)])
        self.attribute_crossformer = nn.ModuleList([nn.MultiheadAttention(512, 8, batch_first=True) for _ in range(2)])
        self.attribute_crossnorm = nn.ModuleList([nn.LayerNorm(512) for _ in range(2)])
        self.attribute_crossnorm1 = nn.LayerNorm(512)

        self.visual_crossformer = nn.ModuleList([nn.MultiheadAttention(512, 8, batch_first=True) for _ in range(2)])
        self.visual_crossnorm = nn.ModuleList([nn.LayerNorm(512) for _ in range(2)])
        self.visual_crossnorm1 = nn.LayerNorm(512)
        #self.attribute_fc = nn.ModuleList([nn.Sequential(OrderedDict([
        #    ("c_fc", nn.Linear(768, 768 * 4)),
        #    ("gelu", QuickGELU()),
        #    ("c_proj", nn.Linear(768 * 4, 768))
        #])) for _ in range(3)])
        #self.attribute_fcnorm = nn.ModuleList([nn.LayerNorm(768) for _ in range(3)])

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
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad=False
        self.text_embedding = clip_model.token_embedding
        self.text_embedding.requires_grad=False
        self.promptlearner = PromptLearner(self.text_embedding)

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
        model_path = '/home/ubuntun/wyq/pretrain/ViT-B-16.pt'
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

        return model

    def forward(self, x=None, cam_label=None, view_label=None, classifer=None, text_feat=None, text_classifier=None):
        
        bn_cls, cls, xproj, xproj_map = self.extract_image(x, cam_label, view_label)
        prompt_text_feat = self.text_encoder(self.promptlearner(), self.promptlearner.tokenized_prompts)
        norm_text_feat = prompt_text_feat / prompt_text_feat.norm(dim=-1, keepdim=True)
        norm_xproj = xproj / xproj.norm(dim=-1, keepdim=True)
        logits = norm_xproj @ norm_text_feat.t()
        topk_indices = logits.topk(128, dim=1).indices
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 512)
        prompt_text_feat = prompt_text_feat.unsqueeze(0).repeat(x.shape[0], 1, 1)
        prompt_text_feat_select = torch.gather(prompt_text_feat, 1, expanded_indices)
        rebuild_feat = self.rebuild_text_feat(prompt_text_feat_select, xproj_map, cls)

        if self.training:

            product = norm_text_feat @ norm_text_feat.t()
            identity = torch.eye(norm_text_feat.shape[0]).to(prompt_text_feat.device)
            orthogonality_loss = torch.norm(product-identity)

            cls_score1 = self.classifier(bn_cls)
            feat = cls / cls.norm(dim=-1, keepdim=True)
            classifer = classifer / classifer.norm(dim=-1, keepdim=True)
            cls_score = (feat @ classifer.t())/ self.temperature

            cls_score2 = self.h_classifier(self.bottleneck1(rebuild_feat))
            
            text_classifier = text_classifier / text_classifier.norm(dim=-1, keepdim=True)
            rebuild_feat = rebuild_feat / rebuild_feat.norm(dim=-1, keepdim=True)
            cls_score3 = (rebuild_feat @ text_classifier.t()) / self.temperature

            return [cls_score, cls_score1, cls_score2, cls_score3], [cls, rebuild_feat], rebuild_feat, orthogonality_loss

        else:
            return cls, torch.cat((cls, rebuild_feat), 1), xproj, rebuild_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        self.logger.info('Loading pretrained model from {}'.format(trained_path))
    
    def update_prompt(self, x=None):
        assert x is not None
        with torch.no_grad():
            bn_cls, cls, xproj, xproj_map = self.extract_image(x)
        prompt_text_feat = self.text_encoder(self.promptlearner(), self.promptlearner.tokenized_prompts)
        rebuild_feat = self.rebuild_text_feat(prompt_text_feat, xproj_map, self.promptlearner)
        
        norm_xproj = xproj / xproj.norm(dim=-1, keepdim=True)
        
        norm_text_feat = prompt_text_feat / prompt_text_feat.norm(dim=-1, keepdim=True)
        norm_rebuild_feat = rebuild_feat / rebuild_feat.norm(dim=-1, keepdim=True)
        return norm_rebuild_feat, norm_xproj
    
    def qmodel_extract(self, x, prompt_text_feat):
        bn_cls, cls, xproj, xproj_map = self.extract_image(x)
        prompt_text_feat = self.text_encoder(self.promptlearner(), self.promptlearner.tokenized_prompts)
        norm_text_feat = prompt_text_feat / prompt_text_feat.norm(dim=-1, keepdim=True)
        norm_xproj = xproj / xproj.norm(dim=-1, keepdim=True)
        logits = norm_xproj @ norm_text_feat.t()
        topk_indices = logits.topk(128, dim=1).indices
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 512)
        prompt_text_feat = prompt_text_feat.unsqueeze(0).repeat(x.shape[0], 1, 1)
        prompt_text_feat_select = torch.gather(prompt_text_feat, 1, expanded_indices)
        rebuild_feat = self.rebuild_text_feat(prompt_text_feat_select, xproj_map, cls)        

        return cls, xproj, rebuild_feat
    
    def extract_image(self, x=None, cam_label=None, view_label=None):
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
        xproj_token = xproj[:, 0]
        bn_cls = self.bottleneck(cls)
        return bn_cls, cls, xproj_token, xproj

    def rebuild_text_feat(self, text_feat, img_feat_map, cls):
        xproj_map = img_feat_map
        
        attr_feat = self.attribute_crossnorm[0](text_feat)
        norm_xproj_map = self.attribute_crossnorm1(xproj_map)
        attr_feat = self.attribute_crossformer[0](attr_feat, norm_xproj_map, norm_xproj_map)[0] + attr_feat

        attr_feat = self.attribute_crossnorm[1](attr_feat)
        attr_feat = self.attribute_crossformer[1](attr_feat, norm_xproj_map, norm_xproj_map)[0] + attr_feat
       
        norm_map_attr_feat = self.visual_crossnorm[0](attr_feat)
        norm_xproj_cls = self.visual_crossnorm1(img_feat_map[:, 0])
        
        visual_cls = norm_xproj_cls.unsqueeze(1)
        visual_cls = self.visual_crossformer[0](visual_cls, norm_map_attr_feat, norm_map_attr_feat)[0] + visual_cls
        visual_cls = self.visual_crossnorm[1](visual_cls)
        visual_cls = self.visual_crossformer[1](visual_cls, norm_map_attr_feat, norm_map_attr_feat)[0] + visual_cls

        #norm_attr_feat = attr_feat / attr_feat.norm(dim=-1, keepdim=True)
        #norm_xproj_cls = xproj_cls / xproj_cls.norm(dim=-1, keepdim=True)
        #logits = norm_xproj_cls.unsqueeze(1) @ norm_attr_feat.transpose(1, 2)
        #combine_feat = logits @ attr_feat
        return visual_cls.squeeze(1)

class PromptLearner(nn.Module):
    def __init__(self, token_embedding):
        super().__init__()
        ctx_init = "A photo of a person"
        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 5
        dtype = torch.float32
        tokenized_prompts = clip.tokenize(ctx_init).cuda()

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        #self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        n_cls_ctx = 20
        n_cls_attr = 256 # 256 - 71 = 185
        self.n_cls_attr = n_cls_attr
        cls_vectors = torch.empty(n_cls_attr, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) # 512x4x512

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx, :])
        self.register_buffer("token_eos", embedding[:, n_ctx+1, :].unsqueeze(1))
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.n_cls_ctx = n_cls_ctx
        tokenized_position = 26
        # tokenized_prompts[:,26] = tokenized_prompts[:,6]
        # tokenized_prompts[:,6] = 0
        self.tokenized_prompts = tokenized_position

    def forward(self):
        prefix = self.token_prefix.expand(self.n_cls_attr, -1, -1)
        eos = self.token_eos.expand(self.n_cls_attr, -1, -1)
        suffix = self.token_suffix.expand(self.n_cls_attr, -1, -1)
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                self.cls_ctx,     # (n_cls, n_ctx, dim)
                eos,
                suffix,
            ],
            dim=1,
        )
        return prompts
