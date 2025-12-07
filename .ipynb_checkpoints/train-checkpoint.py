# 需要安装的包文件
import copy

import torch
import random
import numpy as np
import os
import argparse
import datetime
from config import cfg
import sys

# 本地的包文件
from utils.logger import setup_logger
from Dataloader.make_image_dataloader import make_image_dataloader
from Dataloader.make_video_dataloader import make_video_dataloader
from Models import make_model
from Losses.MakeLoss import make_loss
from solver.MakeOptimizer import make_optimizer
from solver.LrScheduler import WarmupMultiStepLR, CosineLRScheduler
from Processor.TrainProcessor import do_train

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(cfg):

    # 根据时间定义输出文件夹名字
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):      # 创建输出文件夹
        os.mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("ReIDAdapter", cfg.OUTPUT_DIR, if_train=True)          # 定义日志
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))           # 输出加载参数文件的路径
    logger.info("Running with config:\n{}".format(cfg))                          # 输出默认参数
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID                     # 定义需要使用的 GPU id

    # 定义训练和测试的迭代器（数据处理）
    train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num, view_num, train_query_set = make_image_dataloader(cfg)
    # 定义模型
    model = make_model(cfg, num_classes = num_classes, camera_num=camera_num, view_num=view_num)
    q_model = copy.deepcopy(model)

    # 定义损失函数
    loss_function = make_loss(cfg, num_classes)

    # 定义优化器（注意，如果实现冻结某些区域的参数，就可以在这里进行操作）
    optimizer = make_optimizer(cfg,model, logger)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    logger.info("Parameters size: {:.5f}M".format(num_parameters))

    scheduler = CosineLRScheduler(optimizer, t_initial=cfg.SOLVER.MAX_EPOCHS, lr_min=0.002 * cfg.SOLVER.BASE_LR, t_mul= 1.,
                                    decay_rate=0.1, warmup_lr_init=0.01 * cfg.SOLVER.BASE_LR, warmup_t=cfg.SOLVER.WARMUP_EPOCHS,
                                    cycle_limit=1, t_in_epochs=True, noise_range_t=None, noise_pct= 0.67, noise_std= 1., noise_seed=42,)

    # 开始训练
    do_train(
        cfg,
        q_model,
        model,
        train_loader,
        query_loader,
        gallery_loader,
        optimizer,
        scheduler,
        loss_function,
        num_query, args.local_rank,train_query_set
    )


if __name__ == '__main__':

    # 加载参数

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/MSMT17/SAMAdapeter_ReID.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    set_seed(cfg.SOLVER.SEED)

    main(cfg)


