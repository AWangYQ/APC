import logging
import time
from datetime import timedelta

import ipdb

from utils.ranking import ranking

from Processor.inference import inference

def do_inference(cfg,
                 model,
                 q_model,
                 query_loader,
                 gallery_loader,
                 train_loader,
                 train_set, local_rank, sample='yes'):
    
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    model.to(local_rank)  # 将模型由CPU转到GPU
    eval = inference(cfg, num_query)
    start_time = time.monotonic()
    bad_retrieve_results, mAPs = eval.calcultion(cfg, val_loader, model, device, epoch=0)
    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info("Total running time: {}".format(total_time))

    if sample == 'yes':
        logger.info("Bad samples ranking list is generating ...")
        ranking(cfg, bad_retrieve_results, mAPs, model, True)