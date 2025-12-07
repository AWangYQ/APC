import torch
from tqdm import tqdm
import logging

from utils.metrics import R1_mAP_eval

class inference(object):
    def __init__(self, cfg, num_query):
        self.logger = logging.getLogger("ReIDAdapter.inference")
        self.evaluator = R1_mAP_eval(num_query, self.logger, is_cat=cfg.DATASETS.IS_CAT
                            ,max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)  # 定义记录测试数据的函数
    def calcultion(self, cfg, query_loader, gallery_loader, model_list, device, epoch=None):
        self.evaluator.reset()
        if len(model_list) == 2:
            q_model = model_list[0]
            g_model = model_list[1]
            q_model.eval()
            g_model.eval()
            
            for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(tqdm(query_loader)):
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
                    feat = q_model(img, cam_label=camids, view_label=target_view)
                    self.evaluator.update((feat, vid, camid, img_paths))
                    
            for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(tqdm(gallery_loader)):
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
                    feat = g_model(img, cam_label=camids, view_label=target_view)
                    self.evaluator.update((feat, vid, camid, img_paths))
                    
            cmc, mAP, _, _, _, _, _, bad_retrieve_results, mAPs = self.evaluator.compute()
            if epoch != 0:
                self.logger.info("Validation Results - Epoch: {}".format(epoch))
            else:
                self.logger.info("The performance of the pre-trained model:")

            self.logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                
        else:
            model = model_list[0]
            model.eval()

            for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(tqdm(query_loader)):
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
                    self.evaluator.update((feat, vid, camid, img_paths))
                    
            for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(tqdm(gallery_loader)):
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
                    self.evaluator.update((feat, vid, camid, img_paths))

            cmc, mAP, _, _, _, _, _, bad_retrieve_results, mAPs = self.evaluator.compute()
            if epoch != 0:
                self.logger.info("Validation Results - Epoch: {}".format(epoch))
            else:
                self.logger.info("The performance of the pre-trained model:")

            self.logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        return bad_retrieve_results, mAPs