import copy

import torch
import random
import numpy as np
import os
import argparse
import datetime
from config import cfg
import sys

from utils.logger import setup_logger
from Dataloader.make_image_dataloader import make_image_dataloader
from Dataloader.make_video_dataloader import make_video_dataloader
from Models import make_model
from Losses.MakeLoss import make_loss
from solver.MakeOptimizer import make_optimizer, make_prompt_optimizer
from solver.LrScheduler import WarmupMultiStepLR, CosineLRScheduler
from Processor.TrainProcessor import do_train
from Models.ClipBase import PromptLearner

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(cfg):

    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):      
        os.mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("ReIDAdapter", cfg.OUTPUT_DIR, if_train=True)          
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))           
    logger.info("Running with config:\n{}".format(cfg))                          
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID                     

    train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num, view_num, train_query_set = make_image_dataloader(cfg)
    model = make_model(cfg, num_classes = num_classes, camera_num=camera_num, view_num=view_num)
    q_model = copy.deepcopy(model)
    #promptlearner = PromptLearner(model.text_embedding)

    loss_function = make_loss(cfg, num_classes)

    optimizer = make_optimizer(cfg,model, logger)
    #prompt_optimizer = make_prompt_optimizer(cfg, promptlearner, logger)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    logger.info("Parameters size: {:.5f}M".format(num_parameters))

    scheduler = CosineLRScheduler(optimizer, t_initial=cfg.SOLVER.MAX_EPOCHS, lr_min=0.002 * cfg.SOLVER.BASE_LR, t_mul= 1.,
                                    decay_rate=0.1, warmup_lr_init=0.01 * cfg.SOLVER.BASE_LR, warmup_t=cfg.SOLVER.WARMUP_EPOCHS,
                                    cycle_limit=1, t_in_epochs=True, noise_range_t=None, noise_pct= 0.67, noise_std= 1., noise_seed=42,)
    #prompt_scheduler = WarmupMultiStepLR(prompt_optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        q_model,
        model,
     #   promptlearner,
        train_loader,
        query_loader,
        gallery_loader,
        optimizer,
      #  prompt_optimizer,
        scheduler,
       # prompt_scheduler,
        loss_function,
        num_query,0,train_query_set
    )


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/MSMT17/SAMAdapeter_ReID.yml", help="path to config file", type=str
    )
    torch.cuda.set_device(0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    set_seed(cfg.SOLVER.SEED)

    main(cfg)


