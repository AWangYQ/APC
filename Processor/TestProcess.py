import logging
import time
from datetime import timedelta

import ipdb

from Processor.inference import inference

def do_inference(cfg,
                 model,
                 query_loader, gallery_loader,
                 num_query, local_rank, sample='yes'):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    model.to(local_rank)  # 将模型由CPU转到GPU
    eval = inference(cfg, num_query)
    start_time = time.monotonic()
    bad_retrieve_results, mAPs, attr_dict, logit_dict = eval.calcultion(cfg, query_loader, gallery_loader, [model], device, epoch=0)
    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info("Total running time: {}".format(total_time))

    if sample == 'yes':
        logger.info("Bad samples ranking list is generating ...")
        ranking(cfg, bad_retrieve_results, mAPs, model, False, attr_dict, logit_dict)
