# 需要安装的包文件
import os
import argparse
import datetime
from config import cfg
import copy
import torchvision.transforms as T
import torch 
import ipdb
from torch.nn import functional as F

# 本地的包文件
from utils.logger import setup_logger
from Dataloader.make_image_dataloader import make_image_dataloader
from Models import make_model
from Processor.TestProcess import do_inference
from Dataset.image.ImageBase import read_image
from torch.utils.data import DataLoader, Dataset



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

    train_loader, query_loader, gallery_loader, num_query, num_classes, camera_num, view_num, train_query_set = make_image_dataloader(cfg)

    # 定义模型
    model = make_model(cfg, num_classes = num_classes, camera_num=camera_num, view_num=view_num)
    q_model = copy.deepcopy(model)
    model.load_param(cfg.TEST.WEIGHT[0])
    q_model.load_param(cfg.TEST.WEIGHT[1])
    device = "cuda"
    q_model.to(device)
    model.to(device)
    q_model.eval()
    model.eval()
    
    feat_centories = []
    camids = []
    q_pids = []
    for _, (q_img, q_pid, _, camid, _, _) in enumerate(query_loader):
        with torch.no_grad():
            q_img = q_img.to(device)
            q_feat = model(q_img)[0]
            feat_centories.append(q_feat)
            camids.append(camid)
            q_pids.append(torch.tensor(q_pid))
    
    q_pids = torch.cat(q_pids, 0)
    classifier = torch.cat(feat_centories, 0).to(device).detach()
    camids = torch.cat(camids, 0).to(device).detach()
    
    same_camera_score = 0.
    cross_camera_score = 0.
    same_camera = 0.
    cross_camera = 0.
    num_same_camera = 0
    num_cross_camera = 0
    cross_camera_id = 0
    same_camera_id = 0
    
    for n_iter, (img, vid, _, target_cam, target_view, _) in enumerate(gallery_loader):
        with torch.no_grad():

            img = img.to(device)   # 将变量移入GPU
            vid = torch.tensor(vid)
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
            score = model(img, cam_label=target_cam, view_label=target_view, classifer=classifier)[1]  # 变量输入模型
            if n_iter == 0:
                init_id = vid[0]
                mask = init_id == q_pids
                select_camids = camids[mask]
                select_score = score[0][mask]

            for n,i in enumerate(vid):
                if i != init_id:
                    if num_same_camera != 0:
                        same_camera_score = same_camera_score + same_camera / num_same_camera
                        same_camera_id = same_camera_id+1
                    if num_cross_camera != 0:
                        cross_camera_score = cross_camera_score + cross_camera / num_cross_camera
                        cross_camera_id = cross_camera_id + 1
                    
                    mask = init_id == q_pids
                    select_camids = camids[mask]
                    select_score = score[n][mask]    

                    same_camera = 0.
                    cross_camera = 0.
                    num_same_camera = 1
                    num_cross_camera = 1
                else:
                    for k in range(select_score.size(0)):  
                        if select_camids[k] == target_cam[n]:
                            same_camera = same_camera + select_score[k]
                            num_same_camera = num_same_camera + 1
                        elif select_camids[k] != target_cam[n]:
                            cross_camera = cross_camera + select_score[k]
                            num_cross_camera = num_cross_camera + 1
                        else:
                            ipdb.set_trace()
                init_id = i
                
    same_camera_score = same_camera_score / same_camera_id
    cross_camera_score = cross_camera_score / cross_camera_id
    ipdb.set_trace()

    
    
    

def list_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

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