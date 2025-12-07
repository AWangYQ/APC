import logging
import os
import time
from torch.nn import functional as F
import ipdb
import torch
import torch.nn as nn
from torch.cuda import amp
import time
from datetime import timedelta
from tqdm import tqdm
import random
from utils.ranking import ranking
from utils.meter import AverageMeter
from Processor.inference import inference
from Dataset.image.ImageBase import read_image
import traceback
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Models.backbone import clip

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

attribute_list = [
    # gender
    "male",
    "female",
    
    # age
    "child",
    "teenager",
    "adult",
    "elderly person",
    
    # pants
    "person with trousers",
    "person with pants",
    "person with jeans",
    "person with shorts",
    "person with skirt",
    "person with leggings",
    "person with sweatpants",
    "person with cargo pants",
    "person with chinos",
    "person with joggers",
    "person with culottes",
    "person with slacks",
    
    # trousers
    "person with Sweater",
    "person with Hoodie",
    "person with Jacket",
    "person with Coat",
    "person with Blazer",
    "person with Cardigan",
    "person with Peacoat",
    "person with Windbreaker",
    "person with Puffer jacket",
    
    # bag
    "person with Handbag",
    "person with Shoulder Bag",
    "person with Crossbody Bag",
    "person with Backpack",
    "person with Clutch",
    "person with Briefcase",
    "person with Bucket Bag",
    "person with Tote Bag",
    "person with Messenger Bag",
    "person with Cosmetic Bag",
    "person with Luggage",
    # hat
    "person with Sun Hat",
    "person with Flat Cap",
    "person with Fedora",
    "person with Bucket Hat",
    "person with Baseball Cap",
    
    # hair
    "person with short hair",
    "person with medium hair",
    "person with long hair",
    "person with straight hair",
    "person with wavy hair",
    "person with curly hair",
    "person with coily hair",
    
    # earphone
    "person wearing a pair of in-ear earphones with sleek cables",
    "person wearing a pair of over-ear headphones with large, cushioned ear cups",
    
    
    # glasses
    "person wearing a pair of myopia glasses with sleek, black frames",
    "person wearing a pair of hyperopia glasses with elegant, gold-rimmed frames",
    "person wearing a pair of presbyopia glasses with stylish, tortoiseshell frames",
    "person wearing a pair of astigmatism glasses with modern, rectangular frames",
    "person wearing a pair of stylish sunglasses with dark, polarized lenses and aviator-style metal frames",
    
    
    # body
    "person with an hourglass figure, featuring a well-defined waist and balanced proportions between the bust and hips",
    "person with an apple-shaped body, characterized by a fuller midsection and slimmer legs",
    "person with a pear-shaped body, showcasing wider hips and thighs with a narrower upper body",
    "person with a rectangular body shape, where the bust, waist, and hips are relatively similar in width, creating a straight silhouette",
    "person with an inverted triangle body shape, marked by broad shoulders and a narrower waist and hips",
    "slim person, with a lean physique and minimal body fat",
    "person with an average build, featuring balanced proportions and a moderate amount of body fat",
    "curvy person, with pronounced curves and a fuller figure",
    "tall person, with long limbs and an elongated frame",
    "petite person, with a shorter stature and smaller frame",
    "muscular person, showcasing well-defined muscles and a strong physique",
    "person with a softer body type, characterized by less muscle definition and a more rounded appearance",
    "person with evenly distributed body fat, creating a balanced and proportionate look",
    "photo of a person with localized fat distribution, where certain areas like the abdomen, hips, or thighs are more pronouncedq"
]

imagenet_templates = [
   # 'a bad photo of a {}.',
   # 'a photo of many {}.',
   # 'a sculpture of a {}.',
   # 'a photo of the hard to see {}.',
   # 'a low resolution photo of the {}.',
   # 'a rendering of a {}.',
   # 'graffiti of a {}.',
   # 'a bad photo of the {}.',
   # 'a cropped photo of the {}.',
   # 'a tattoo of a {}.',
   # 'the embroidered {}.',
   # 'a photo of a hard to see {}.',
   # 'a bright photo of a {}.',
   # 'a photo of a clean {}.',
   # 'a photo of a dirty {}.',
   # 'a dark photo of the {}.',
   # 'a drawing of a {}.',
   # 'a photo of my {}.',
   # 'the plastic {}.',
   # 'a photo of the cool {}.',
   # 'a close-up photo of a {}.',
   # 'a black and white photo of the {}.',
   # 'a painting of the {}.',
   # 'a painting of a {}.',
   # 'a pixelated photo of the {}.',
   # 'a sculpture of the {}.',
   # 'a bright photo of the {}.',
   # 'a cropped photo of a {}.',
   # 'a plastic {}.',
   # 'a photo of the dirty {}.',
   # 'a jpeg corrupted photo of a {}.',
   # 'a blurry photo of the {}.',
   # 'a photo of the {}.',
   # 'a good photo of the {}.',
   # 'a rendering of the {}.',
   # 'a {} in a video game.',
   # 'a photo of one {}.',
   # 'a doodle of a {}.',
   # 'a close-up photo of the {}.',
   'a photo of a {}.',
   # 'the origami {}.',
   # 'the {} in a video game.',
   # 'a sketch of a {}.',
   # 'a doodle of the {}.',
   # 'a origami {}.',
   # 'a low resolution photo of a {}.',
   # 'the toy {}.',
   # 'a rendition of the {}.',
   # 'a photo of the clean {}.',
   # 'a photo of a large {}.',
   # 'a rendition of a {}.',
   # 'a photo of a nice {}.',
   # 'a photo of a weird {}.',
   # 'a blurry photo of a {}.',
   # 'a cartoon {}.',
   # 'art of a {}.',
   # 'a sketch of the {}.',
   # 'a embroidered {}.',
   # 'a pixelated photo of a {}.',
   # 'itap of the {}.',
   # 'a jpeg corrupted photo of the {}.',
   # 'a good photo of a {}.',
   # 'a plushie {}.',
   # 'a photo of the nice {}.',
   # 'a photo of the small {}.',
   # 'a photo of the weird {}.',
   # 'the cartoon {}.',
   # 'art of the {}.',
   # 'a drawing of the {}.',
   # 'a photo of the large {}.',
   # 'a black and white photo of a {}.',
   # 'the plushie {}.',
   # 'a dark photo of a {}.',
   # 'itap of a {}.',
   # 'graffiti of the {}.',
   # 'a toy {}.',
   # 'itap of my {}.',
   # 'a photo of a cool {}.',
   # 'a photo of a small {}.',
   # 'a tattoo of the {}.',
]

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 0.07
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        loss = - mean_log_prob_pos.mean()

        return loss

def norm(tensor, cfg):
    if cfg.MODEL.NORM == 'gen':
        tensor = (1 + tensor) / 2
        return tensor / tensor.sum(-1, keepdim=True)
    elif cfg.MODEL.NORM == 'softmax':
        return F.softmax(tensor, -1)

def do_train(cfg,
             q_model,
             model,
             train_loader,
             query_loader,
             gallery_loader,
             optimizer,
             scheduler,
             loss_function,
             num_query, local_rank, train_query_set):

    #model.load_param('/home/ubuntun/wyq/output/2024-11-05_17-07-53/q_model_80.pth')
    #param_dict = torch.load('/home/ubuntun/wyq/output/2024-11-05_17-07-53/prompt_learner_80.pth')
    #for i in param_dict:
    #    prompt_learner.state_dict()[i].copy_(param_dict[i])

    log_period = cfg.SOLVER.LOG_PERIOD  # 每多少次迭代输出一次日志
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD   # 每多少次迭代保存模型
    eval_period = cfg.SOLVER.EVAL_PERIOD     # 每多少次迭代进行测试

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS    # 最大迭代次数

    logger = logging.getLogger("ReIDAdapter.train")
    logger.info('start training')

    model.to(local_rank)  # 将模型由CPU转到GPU
    q_model.to(local_rank)

    loss_meter = AverageMeter()  # 定义记录损失函数变化的函数
    acc_meter = AverageMeter()   # 定义记录准确度变化的函数
    align_loss_meter = AverageMeter()
    acc_meter0 = AverageMeter()
    
    scaler = amp.GradScaler()   # 防止反向传播中的数值不稳定性问题，梯度爆炸，梯度消失
    eval = inference(cfg, num_query)
    #q_schedule = cosine_scheduler(0.96, 1, 120, len(train_loader)//50)
    #text = []
    #for attr in attribute_list:
    #    attr_texts = [template.format(attr) for template in imagenet_templates]
    #    text.extend(attr_texts)
    #text = clip.tokenize(text).to(device)

    #with torch.no_grad():
    #    text_features = model.text_encoder(text=text)

    #pretrain_instances_logits = torch.load(cfg.MODEL.LOGITS_FILE2)
    #pretrain_class_logits = torch.load(cfg.MODEL.LOGITS_FILE1)
    #pre_log_matrix = []
    #for _, tensor in pretrain_class_logits.items():
    #    pre_log_matrix.append(tensor)
    #pre_log_matrix = torch.stack(pre_log_matrix, 0).float().cuda()
    #eval.calcultion(cfg, query_loader, gallery_loader, [model], device, epoch=0)
    #import ipdb; ipdb.set_trace()
    all_start_time = time.monotonic()  # 开始时间
    xent = SupConLoss(local_rank)
    q_model.eval()

    feat_centories = []
    xproj_centories = []
    text_centories = []
    pids = []
    
    with torch.no_grad():
        prompt_text_feat = model.text_encoder(model.promptlearner(), model.promptlearner.tokenized_prompts)

    for _, (q_img, q_pid) in enumerate(train_query_set[0]):
        with torch.no_grad(): 
            q_img = q_img.to(device)
            q_feat, xproj, text_feat = q_model.qmodel_extract(q_img, prompt_text_feat)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
            xproj = xproj / xproj.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            feat_centories.append(q_feat)
            xproj_centories.append(xproj)
            text_centories.append(text_feat)
            pids.append(q_pid)

    classifer = torch.cat(feat_centories, 0).to(device).detach()
    xproj_classifer = torch.cat(xproj_centories, 0).to(device).detach()
    text_classifer = torch.cat(text_centories, 0).to(device).detach()
    pids = torch.cat(pids, 0).to(device)

    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()  # 重置函数内的静态参数
        acc_meter.reset()
        acc_meter0.reset()
        align_loss_meter.reset()
        labels = []
        text_feats = []
        img_feats = []

        model.train()      # 设置模型为训练状态
        for n_iter, (img, vid, target_cam, target_view, img_name) in enumerate(train_loader):
            #class_logits = pre_log_matrix[vid]
            #instance_logits = [pretrain_instances_logits[name].to(device) for name in img_name]
            #instance_logits = torch.stack(instance_logits, 0)
            optimizer.zero_grad()  # 清零梯度
            img = img.to(device)   # 将变量移入GPU
            target = vid.to(device)
            # 计算SIE 信息
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                #combine_feat, xproj, logits_0 = q_model.update_prompt(img, prompt_learner)
                #align_loss = xent(combine_feat, xproj_classifer, target, pids)
                #score_0 = (combine_feat @ text_classifer.t()) / 0.007
                #ce_loss = F.cross_entropy(score_0, target)
                #align_loss = align_loss + ce_loss        
 
                score, feat, combine_feat, orthogonality_loss = model(img, cam_label=target_cam, view_label=target_view, classifer=classifer,text_classifier = text_classifer) 
                #align_loss = xent(combine_feat, xproj_classifer, target, pids)
                align_loss = 0.1 * orthogonality_loss
                # pred_logits = pred_logits.reshape(-1, 71, 80).mean(2)
                # pred_logits = pred_logits / pred_logits.norm(dim=-1, keepdim=True)
                # class_logits = class_logits / class_logits.norm(dim=-1, keepdim=True)
                # align_loss = xent(pred_logits, class_logits, vid, vid)
                # labels.append(vid)
                # text_feats.append(feat[1].cpu())
                # img_feats.append(feat[0].cpu())
                # pred_gender_logits = norm(pred_logits[:,:2],cfg)
                # gender_logits = norm(logits[:,:2],cfg)
                # kl1 = F.kl_div(torch.log(pred_gender_logits), gender_logits.float().cuda(), reduction='batchmean')

                # pred_age_logits = norm(pred_logits[:,2:6],cfg)
                # age_logits = norm(logits[:,2:6],cfg)
                # kl2 = F.kl_div(torch.log(pred_age_logits), age_logits.float().cuda(), reduction='batchmean')

                # pred_trousers_logits =  norm(pred_logits[:,6:46],cfg)
                # trousers_logits =  norm(logits[:,6:46],cfg)
                # kl3 = F.kl_div(torch.log(pred_trousers_logits), trousers_logits.float().cuda(), reduction='batchmean')

                # pred_coat_logits =  norm(pred_logits[:,46:94],cfg)
                # coat_logits =  norm(logits[:,46:94],cfg)
                # kl4 = F.kl_div(torch.log(pred_coat_logits), coat_logits.float().cuda(), reduction='batchmean')

                # pred_hair_logits =  norm(pred_logits[:,94:104],cfg)
                # hair_logits =  norm(logits[:,94:104],cfg)
                # kl5 = F.kl_div(torch.log(pred_hair_logits), hair_logits.float().cuda(), reduction='batchmean')

                # pred_bag_logits =  norm(pred_logits[:,104:121],cfg)
                # bag_logits =  norm(logits[:,104:121],cfg)
                # kl6 = F.kl_div(torch.log(pred_bag_logits), bag_logits.float().cuda(), reduction='batchmean')

                # pred_shoes_logits =  norm(pred_logits[:,121:145],cfg)
                # shoes_logits =  norm(logits[:,121:145],cfg)
                # kl7 = F.kl_div(torch.log(pred_shoes_logits), shoes_logits.float().cuda(), reduction='batchmean')

                # pred_hat_logits =  norm(pred_logits[:,145:154],cfg)
                # hat_logits =  norm(logits[:,145:154],cfg)
                # kl8 = F.kl_div(torch.log(pred_hat_logits), hat_logits.float().cuda(), reduction='batchmean')

                # pred_glasses_logits =  norm(pred_logits[:,154:],cfg)
                # glasses_logits =  norm(logits[:,154:],cfg)
                # kl9 = F.kl_div(torch.log(pred_glasses_logits), glasses_logits.float().cuda(), reduction='batchmean')
                # kl_loss = kl1 + kl2 + kl3 + kl4 + kl5 + kl6 + kl7 + kl8 + kl9
                loss = loss_function(score, feat, target, target_cam)                    # 计算损失函数

            scaler.scale(loss+align_loss).backward()
            scaler.step(optimizer)          # 加载优化器
            scaler.update()                 # 更新参数
            #state_dict = q_model.state_dict()
            #for name, param in model.named_parameters():
            #    if not (state_dict[name] == param).any():
            #        print(name)
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()  
            else:
                acc = (score.max(1)[1] == target).float().mean()     # 计算训练集数据的训练精度
            acc0 = (score[2].max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])     # 在记录函数中更新每次迭代的这些参数
            align_loss_meter.update(align_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)                         # 因为在前面计算acc中已经计算了平均，所以这里不需要了
            acc_meter0.update(acc0, 1)
            torch.cuda.synchronize()                         # 为了保证CPU和GPU的运算一致
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, align loss: {:.3f}, Acc0: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, align_loss_meter.avg, acc_meter0.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                with torch.no_grad():
                    prompt_text_feat = model.text_encoder(model.promptlearner(), model.promptlearner.tokenized_prompts)
                q_model.eval()
                feat_centories = []
                xproj_centories = []
                text_centories = []
                for _, (q_img, q_pid) in enumerate(train_query_set[(epoch-1)* 9 + (n_iter+1) // 50]):
                    with torch.no_grad():
                        q_img = q_img.to(device)
                        q_feat, xproj, text_feat = q_model.qmodel_extract(q_img, prompt_text_feat)
                        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
                        xproj = xproj / xproj.norm(dim=-1, keepdim=True)
                        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                        feat_centories.append(q_feat)
                        xproj_centories.append(xproj)
                        text_centories.append(text_feat)

                classifer = torch.cat(feat_centories, 0).to(device).detach()
                xproj_classifer = torch.cat(xproj_centories, 0).to(device).detach()
                text_classifer = torch.cat(text_centories, 0).to(device).detach()
           
                with torch.no_grad():
                    #m = q_schedule[(epoch-1)* 9 + (n_iter+1) // 50]  # momentum parameter
                    for param_q, param_k in zip(model.parameters(), q_model.parameters()):
                        param_k.data.mul_(0.996).add_((1 - 0.996) * param_q.detach().data)
            
            #with torch.no_grad():
            #    q_model.train()
            #    _, _, pred_logits = q_model(img, cam_label=target_cam, view_label=target_view, classifer=classifier, text_feat = text_features, attr_classifier = attr_classifier)
            #    #pred_logits = pred_logits.reshape(-1, 4, 71)
            #    pred_logits = pred_logits.reshape(-1, 71, 80).mean(2)
            #    pred_logits = pred_logits.reshape(-1, 4, 71).mean(1)
            #    index = vid[::4]
            #    pre_log_matrix[index] = pre_log_matrix[index] * 0.8 + 0.2 * pred_logits 
            
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)     # 统计完成一次训练迭代需要的时间。
        scheduler.step(epoch)  # 更新学习率，这个需要放在epoch的循环内，因为学习率是一整个迭代完成了，再更新

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))  # 输出训练的情况，分别包括迭代次数，迭代时间

        # 保存模型
        if epoch % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            torch.save(q_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'q_model' + '_{}.pth'.format(epoch)))

        # 测试模型
        if epoch % eval_period == 0 or epoch == 1 or (epoch <= 80 and epoch >=60):
            #logger.info("===============================")
            #logger.info("Testing the gallery model ....")
            #query_img_paths, gallery_img_paths, _, _ = eval.calcultion(cfg, query_loader, gallery_loader, [model], device, epoch)
            logger.info("===============================")
            logger.info("Testing the query model ......")
            eval.calcultion(cfg, query_loader, gallery_loader, [q_model], device, epoch)
            #logger.info("===============================")
            #logger.info("Testing cross model ......")
            #eval.calcultion(cfg, query_loader, gallery_loader, [q_model, model], device, epoch)
            
        all_end_time = time.monotonic()
        total_time = timedelta(seconds=all_end_time - all_start_time)
        logger.info("Total running time: {}".format(total_time))

    if cfg.TEST.REPORT_SAMPLE:
        logger.info("Bad samples ranking list is generating ...")
        ranking(cfg, query_img_paths, gallery_img_paths, model, False)

