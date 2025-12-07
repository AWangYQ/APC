# 需要安装的包文件
import os
import argparse
import datetime
from config import cfg

# 本地的包文件
from utils.logger import setup_logger
from Dataloader.make_image_dataloader import make_image_dataloader
from Models import make_model
from Processor.TestProcess import do_inference
from Models.SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator


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
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 定义训练和测试的迭代器（数据处理）
    train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num, view_num, train_query_set= make_image_dataloader(cfg)

    # 定义模型
    model = make_model(cfg, num_classes = num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    do_inference(cfg,
                 model,
                 query_loader, gallery_loader,
                 num_query, args.local_rank)


if __name__ == '__main__':

    # 加载参数

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/Mars/VitBase.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    main(cfg)
