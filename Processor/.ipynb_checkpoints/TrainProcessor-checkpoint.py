import logging
import os
import time

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

    scaler = amp.GradScaler()   # 防止反向传播中的数值不稳定性问题，梯度爆炸，梯度消失
    eval = inference(cfg, num_query)
    q_schedule = cosine_scheduler(0.96, 1, 120, len(train_loader)//50)

    if cfg.TEST.ZERO_SHOT:
        eval.calcultion(cfg, val_loader, model, device, epoch=0)

    all_start_time = time.monotonic()  # 开始时间
    
    q_model.eval()
    feat_centories = []
    for _, (q_img, q_pid) in enumerate(train_query_set[0]):
        with torch.no_grad():
            q_img = q_img.to(device)
            q_feat = q_model(q_img)
            feat_centories.append(q_feat)

    classifier = torch.cat(feat_centories, 0).to(device).detach()

    # 开始训练
    for epoch in range(1, epochs+1):
        start_time = time.time()
        loss_meter.reset()  # 重置函数内的静态参数
        acc_meter.reset()

        model.train()      # 设置模型为训练状态
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):

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
                score, feat = model(img, cam_label=target_cam, view_label=target_view, classifer=classifier)  # 变量输入模型
                loss = loss_function(score, feat, target, target_cam)                        # 计算损失函数

            scaler.scale(loss).backward()   # 在scaler的保护下，反传梯度
            scaler.step(optimizer)          # 加载优化器
            scaler.update()                 # 更新参数

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()  # 如果用了多监督，计算这些特征向量的平均精读。当然，如果需要计算某一个特征向量的精读，可以直接指定。
            else:
                acc = (score.max(1)[1] == target).float().mean()     # 计算训练集数据的训练精度

            loss_meter.update(loss.item(), img.shape[0])     # 在记录函数中更新每次迭代的这些参数
            acc_meter.update(acc, 1)                         # 因为在前面计算acc中已经计算了平均，所以这里不需要了

            torch.cuda.synchronize()                         # 为了保证CPU和GPU的运算一致
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                
                q_model.eval()
                feat_centories = []
                for _, (q_img, q_pid) in enumerate(train_query_set[(epoch-1)* 9 + (n_iter+1) // 50]):
                    with torch.no_grad():
                        q_img = q_img.to(device)
                        q_feat = q_model(q_img)
                        feat_centories.append(q_feat)

                classifier = torch.cat(feat_centories, 0).to(device).detach()

                with torch.no_grad():
                    m = q_schedule[(epoch-1)* 9 + (n_iter+1) // 50]  # momentum parameter
                    for param_q, param_k in zip(model.parameters(), q_model.parameters()):
                        param_k.data.mul_(0.996).add_((1 - 0.996) * param_q.detach().data)

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)     # 统计完成一次训练迭代需要的时间。

        scheduler.step(epoch)  # 更新学习率，这个需要放在epoch的循环内，因为学习率是一整个迭代完成了，再更新

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))  # 输出训练的情况，分别包括迭代次数，迭代时间

        # 保存模型
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            torch.save(q_model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'q_model' + '_{}.pth'.format(epoch)))

        # 测试模型
        if epoch % eval_period == 0 or epoch == 1:
            logger.info("===============================")
            logger.info("Testing the gallery model ....")
            query_img_paths, gallery_img_paths = eval.calcultion(cfg, query_loader, gallery_loader, [model], device, epoch)
            logger.info("===============================")
            logger.info("Testing the query model ......")
            eval.calcultion(cfg, query_loader, gallery_loader, [q_model], device, epoch)
            logger.info("===============================")
            logger.info("Testing cross model ......")
            eval.calcultion(cfg, query_loader, gallery_loader, [q_model, model], device, epoch)
            
        all_end_time = time.monotonic()
        total_time = timedelta(seconds=all_end_time - all_start_time)
        logger.info("Total running time: {}".format(total_time))

    if cfg.TEST.REPORT_SAMPLE:
        logger.info("Bad samples ranking list is generating ...")
        ranking(cfg, query_img_paths, gallery_img_paths, model, False)

