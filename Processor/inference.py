import torch
from tqdm import tqdm
import logging
from Models.backbone import clip
import torch.nn.functional as F
from utils.metrics import R1_mAP_eval
attribute_list = [
    "A photo of a male.",
    "A photo of a female.",
    "A photo of a child.",
    "A photo of a teenager.",
    "A photo of a adult.",
    "A photo of a old.",
    "A photo of a person with black pure trousers.",
    "A photo of a person with black strides.",
    "A photo of a person with black design trousers.",
    "A photo of a person with black joint trousers.",
    "A photo of a person with black lattice trousers.",
    "A photo of a person with white pure trousers.",
    "A photo of a person with white strides.",
    "A photo of a person with white design trousers.",
    "A photo of a person with white joint trousers.",
    "A photo of a person with white lattice trousers.",
    "A photo of a person with gray pure trousers.",
    "A photo of a person with gray strides.",
    "A photo of a person with gray design trousers.",
    "A photo of a person with gray joint trousers.",
    "A photo of a person with gray lattice trousers.",
    "A photo of a person with red pure trousers.",
    "A photo of a person with red strides.",
    "A photo of a person with red design trousers.",
    "A photo of a person with red joint trousers.",
    "A photo of a person with red lattice trousers.",
    "A photo of a person with yellow pure trousers.",
    "A photo of a person with yellow strides.",
    "A photo of a person with yellow design trousers.",
    "A photo of a person with yellow joint trousers.",
    "A photo of a person with yellow lattice trousers.",
    "A photo of a person with blue pure trousers.",
    "A photo of a person with blue strides.",
    "A photo of a person with blue design trousers.",
    "A photo of a person with blue joint trousers.",
    "A photo of a person with blue lattice trousers.",
    "A photo of a person with green pure trousers.",
    "A photo of a person with green strides.",
    "A photo of a person with green design trousers.",
    "A photo of a person with green joint trousers.",
    "A photo of a person with green lattice trousers.",
    "A photo of a person with purple pure trousers.",
    "A photo of a person with purple strides.",
    "A photo of a person with purple design trousers.",
    "A photo of a person with purple joint trousers.",
    "A photo of a person with purple lattice trousers.",
    "A photo of a person with black jacket.",
    "A photo of a person with black sweater.",
    "A photo of a person with black long coat.",
    "A photo of a person with black shirt.",
    "A photo of a person with black dress.",
    "A photo of a person with black business suit.",
    "A photo of a person with white jacket.",
    "A photo of a person with white sweater.",
    "A photo of a person with white long coat.",
    "A photo of a person with white shirt.",
    "A photo of a person with white dress.",
    "A photo of a person with white business suit.",
    "A photo of a person with gray jacket.",
    "A photo of a person with gray sweater.",
    "A photo of a person with gray long coat.",
    "A photo of a person with gray shirt.",
    "A photo of a person with gray dress.",
    "A photo of a person with gray business suit.",
    "A photo of a person with red jacket.",
    "A photo of a person with red sweater.",
    "A photo of a person with red long coat.",
    "A photo of a person with red shirt.",
    "A photo of a person with red dress.",
    "A photo of a person with red business suit.",
    "A photo of a person with yellow jacket.",
    "A photo of a person with yellow sweater.",
    "A photo of a person with yellow long coat.",
    "A photo of a person with yellow shirt.",
    "A photo of a person with yellow dress.",
    "A photo of a person with yellow business suit.",
    "A photo of a person with blue jacket.",
    "A photo of a person with blue sweater.",
    "A photo of a person with blue long coat.",
    "A photo of a person with blue shirt.",
    "A photo of a person with blue dress.",
    "A photo of a person with blue business suit.",
    "A photo of a person with green jacket.",
    "A photo of a person with green sweater.",
    "A photo of a person with green long coat.",
    "A photo of a person with green shirt.",
    "A photo of a person with green dress.",
    "A photo of a person with green business suit.",
    "A photo of a person with purple jacket.",
    "A photo of a person with purple sweater.",
    "A photo of a person with purple long coat.",
    "A photo of a person with purple shirt.",
    "A photo of a person with purple dress.",
    "A photo of a person with purple business suit.",
    "A photo of a bald person.",
    "A photo of a person with long black hair.",
    "A photo of a person with short black hair.",
    "A photo of a person with thick black hair.",
    "A photo of a person with long brown hair.",
    "A photo of a person with short brown hair.",
    "A photo of a person with thick brown hair.",
    "A photo of a person with long red hair.",
    "A photo of a person with short red hair.",
    "A photo of a person with thick red hair.",
    "A photo of a person with black shoulder bag.", 
    "A photo of a person with white shoulder bag.",
    "A photo of a person with gray shoulder bag.",
    "A photo of a person with red shoulder bag.",
    "A photo of a person with yellow shoulder bag.",
    "A photo of a person with blue shoulder bag.",
    "A photo of a person with green shoulder bag.",
    "A photo of a person with purple shoulder bag.",
    "A photo of a person with black backpack.", 
    "A photo of a person with white backpack.",
    "A photo of a person with gray backpack.",
    "A photo of a person with red backpack.",
    "A photo of a person with yellow backpack.",
    "A photo of a person with blue backpack.",
    "A photo of a person with green backpack.",
    "A photo of a person with purple backpack.",
    "A photo of a person without bag.",
    "A photo of a person with black leather shoes.", 
    "A photo of a person with white leather shoes.",
    "A photo of a person with gray leather shoes.",
    "A photo of a person with red leather shoes.",
    "A photo of a person with yellow leather shoes.",
    "A photo of a person with blue leather shoes.",
    "A photo of a person with green leather shoes.",
    "A photo of a person with purple leather shoes.",
    "A photo of a person with black boots.", 
    "A photo of a person with white boots.",
    "A photo of a person with gray boots.",
    "A photo of a person with red boots.",
    "A photo of a person with yellow boots.",
    "A photo of a person with blue boots.",
    "A photo of a person with green boots.",
    "A photo of a person with purple boots.",
    "A photo of a person with black walking shoes.", 
    "A photo of a person with white walking shoes.",
    "A photo of a person with gray walking shoes.",
    "A photo of a person with red walking shoes.",
    "A photo of a person with yellow walking shoes.",
    "A photo of a person with blue walking shoes.",
    "A photo of a person with green walking shoes.",
    "A photo of a person with purple walking shoes.",
    "A photo of a person with black hat.", 
    "A photo of a person with white hat.",
    "A photo of a person with gray hat.",
    "A photo of a person with red hat.",
    "A photo of a person with yellow hat.",
    "A photo of a person with blue hat.",
    "A photo of a person with green hat.",
    "A photo of a person with purple hat.",
    "A photo of a person without hat.",
    "A photo of a person with glasses.",
    "A photo of a person without glasses.",
]
gender_attribute = attribute_list[:2]
age_attribute = attribute_list[2:6]
trousers_attribute = attribute_list[6:46]
coat_attribute = attribute_list[46:94]
hair_attribute = attribute_list[94:104]
bag_attribute = attribute_list[104:121]
shoes_attribute = attribute_list[121:145]
hat_attribute = attribute_list[145:154]
glasses_attribute = attribute_list[154:]

def record_pic_attr(text_feat, xproj, img_names, instance_attr_dict, instance_logit_dict):
    logits_per_image = xproj @ text_feat.t()
    
    for index, name in enumerate(img_names):
        gender_logit= logits_per_image[index][:2]
        age_logit = logits_per_image[index][2:6]
        trousers_logit = logits_per_image[index][6:46]
        coat_logit = logits_per_image[index][46:94]
        hair_logit = logits_per_image[index][94:104]
        bag_logit = logits_per_image[index][104:121]
        shoes_logit = logits_per_image[index][121:145]
        hat_logit = logits_per_image[index][145:154]
        glasses_logit = logits_per_image[index][154:]

        class_attribute = []
        class_attribute.append(gender_attribute[torch.argmax(gender_logit)])
        class_attribute.append(age_attribute[torch.argmax(age_logit)])
        class_attribute.append(trousers_attribute[torch.argmax(trousers_logit)])
        class_attribute.append(coat_attribute[torch.argmax(coat_logit)])
        class_attribute.append(hair_attribute[torch.argmax(hair_logit)])
        class_attribute.append(bag_attribute[torch.argmax(bag_logit)])
        class_attribute.append(shoes_attribute[torch.argmax(shoes_logit)])
        class_attribute.append(hat_attribute[torch.argmax(hat_logit)])
        class_attribute.append(glasses_attribute[torch.argmax(glasses_logit)])
        instance_attr_dict[name] = class_attribute
        instance_logit_dict[name] = logits_per_image[index]

    return instance_attr_dict, instance_logit_dict

class inference(object):
    def __init__(self, cfg, num_query):
        self.logger = logging.getLogger("ReIDAdapter.inference")
        self.evaluator = R1_mAP_eval(num_query, self.logger, is_cat=cfg.DATASETS.IS_CAT
                            ,max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # 定义记录测试数据的函数
    def calcultion(self, cfg, query_loader, gallery_loader, model_list, device, epoch=None, text_feat=None):
        self.evaluator.reset()
        if len(model_list) == 2:
            pass 
        else:
            model = model_list[0]
            model.eval()
            #text = clip.tokenize(attribute_list).to(device)
            #with torch.no_grad():
            #    text_feat = model.text_encoder(text)
            #text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            #assert text_feat is not None
            instance_attr_dict = {}
            instance_logits_dict = {}
            for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(query_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else:
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else:
                        target_view = None
                    feat = model(img, cam_label=camids, view_label=target_view)
                    #proj_feat = feat[1]
                    #proj_feat = proj_feat / proj_feat.norm(dim=-1, keepdim=True)
                    #instance_attr_dict, instance_logits_dict = record_pic_attr(text_feat, proj_feat, img_paths, instance_attr_dict, instance_logits_dict)
                    self.evaluator.update((feat[1], vid, camid, img_paths))
                    
            for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(gallery_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else:
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else:
                        target_view = None
                    feat = model(img, cam_label=camids, view_label=target_view)
                    #proj_feat = feat[1]
                    #proj_feat = proj_feat / proj_feat.norm(dim=-1, keepdim=True)
                    #instance_attr_dict, instance_logits_dict = record_pic_attr(text_feat, proj_feat, img_paths, instance_attr_dict, instance_logits_dict)
                    self.evaluator.update((feat[1], vid, camid, img_paths))
            cmc, mAP, _, _, _, _, _, bad_retrieve_results, mAPs = self.evaluator.compute()
            if epoch != 0:
                self.logger.info("Validation Results - Epoch: {}".format(epoch))
            else:
                self.logger.info("The performance of the pre-trained model:")

            self.logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            
        return bad_retrieve_results, mAPs, instance_attr_dict, instance_logits_dict
