import os.path as osp
import os
import cv2
import ipdb
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, EigenGradCAM, FullGrad, EigenCAM
from tqdm import tqdm
import torch.nn.functional as F
import torch
def reshape_transform(tensor, height=16, width=8):
    if isinstance(tensor, tuple):
        if tensor[0].size(0) > tensor[0].size(1):
            tensor = tensor[0].transpose(0,1)
        else:
            tensor = tensor[0]

    if tensor.size(1) > 128:
        tensor = tensor[:,-128:,:].reshape(1,height, width, 768)
    elif tensor.size(1) == 128:
        tensor = tensor.reshape(1,height, width, 768)
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result

def ranking(cfg, bad_retrieve_results, mAPs, model, show_cam, attr_dict, logit_dict):
    dst = osp.join(cfg.OUTPUT_DIR, 'bad retrieve samples')
    root = osp.join(cfg.DATASETS.ROOT_DIR, 'MSMT17_V1', 'test')
    if not os.path.exists(dst):
        os.mkdir(dst)
    # os.makedirs(dst)

    GRID_SPACING = 10
    QUERY_EXTRA_SPACING = 90
    BW = 5  # border width
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    topk = 15
    height, width = cfg.INPUT.SIZE_TRAIN

    if show_cam:
        target_layers1 = [model.image_encoder.transformer.resblocks[-1].attn]
        # target_layers2 = [model.sam_ImageEncoder.blocks[-1].attn]
        # target_layers3 = [model.FusionDecoder1.DecoderSelfAttention]

        cam1 = EigenCAM(model=model, target_layers=target_layers1, use_cuda=True, reshape_transform=reshape_transform)
        # cam2 = EigenCAM(model=model, target_layers=target_layers2, use_cuda=True, reshape_transform=reshape_transform)
        # cam3 = EigenCAM(model=model, target_layers=target_layers3, use_cuda=True, reshape_transform=reshape_transform)

    retrieve_items = bad_retrieve_results.items()
    num = 0
    for q_name, g_names in tqdm(retrieve_items):
        # query_img_paths, gallery_img_paths = bad_retrieve_results[q_idx]
        q_logit = logit_dict[q_name]
        qpid = q_name.split('_')[0]
        q_attr = attr_dict[q_name]
        q_path = osp.join(root, qpid, q_name)
        qimg = cv2.imread(q_path)
        qimg = cv2.resize(qimg, (width, height))
        if show_cam:
            cam_image1 = PaintingHeatmap(cam1, qimg)
            # cam_image2 = PaintingHeatmap(cam2, qimg)
            # cam_image3 = PaintingHeatmap(cam3, qimg)
        qimg = cv2.copyMakeBorder(
            qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        # resize twice to ensure that the border width is consistent across images
        qimg = cv2.resize(qimg, (width, height))
        num_cols = topk + 1
        if show_cam:
            grid_img = 255 * np.ones(
                (
                    3 * height + 1*GRID_SPACINGi+100,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
        else:
            grid_img = 255 * np.ones(
                (
                    height + 3*GRID_SPACING + 100,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
        grid_img[:height, :width, :] = qimg
        if show_cam:
            grid_img[height+GRID_SPACING:2*height+GRID_SPACING, :width, :] = cam_image1
            # grid_img[height*2 + GRID_SPACING*2:3 * height+2*GRID_SPACING, :width, :] = cam_image2
            # grid_img[height*2 + GRID_SPACING*2:, :width, :] = cam_image3

        rank_idx = 1
        g_attr = []
        qg_kl = []
        for g_name in g_names:
            g_logit = logit_dict[g_name]
            gpid = g_name.split('_')[0]
            g_attr.append(attr_dict[g_name])
            g_path = osp.join(root, gpid, g_name)
            matched = gpid == qpid
            border_color = GREEN if matched else RED
            gimg = cv2.imread(g_path)
            gimg = cv2.resize(gimg, (width, height))
            if show_cam:
                cam_gimg1 = PaintingHeatmap(cam1, gimg)
                # cam_gimg2 = PaintingHeatmap(cam2, gimg)
                # cam_gimg3 = PaintingHeatmap(cam3, gimg)
            gimg = cv2.copyMakeBorder(
                gimg,
                BW,
                BW,
                BW,
                BW,
                cv2.BORDER_CONSTANT,
                value=border_color
                # value = (0, 0, 0)
            )
            gimg = cv2.resize(gimg, (width, height))
            start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
            end = (
                          rank_idx + 1
                  ) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
            grid_img[:height, start:end, :] = gimg
            if show_cam:
                grid_img[height + GRID_SPACING:2 * height + GRID_SPACING, start:end, :] = cam_gimg1
                # grid_img[height * 2 + GRID_SPACING * 2:3 * height + 2 * GRID_SPACING,  start:end, :] = cam_gimg2
                # grid_img[height * 2 + GRID_SPACING * 2:,  start:end, :] = cam_gimg3
            rank_idx += 1
            
            qg_kl.append((q_logit/q_logit.norm(dim=-1, keepdim=True))@( g_logit / g_logit.norm(dim=-1, keepdim=True)).t())

        caption = 'mAP is {}'.format(mAPs[num])
        text_position = (10, 30)  # 文本位置坐标
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        font_scale = 0.5  # 字体大小
        font_color = (0, 0, 255)  # 字体颜色，BGR 格式
        thickness = 2  # 字体线条粆
        cv2.putText(grid_img, caption, text_position, font, font_scale, font_color, thickness)
        caption1 = 'query attribute is {}'.format(q_attr)
        cv2.putText(grid_img, caption1, (10, 270), font, font_scale, font_color, thickness)
        cv2.imwrite(osp.join(dst, q_name + '.jpg'), grid_img)
        file_name = osp.join(dst, q_name + '.txt')
        with open(file_name, 'w') as file:
            file.write(caption1)
            file.write('\n')
            file.write(str(qg_kl))
            file.write('\n')
            file.write('gallery attribute is :')
            file.write('\n')
            for g_a in g_attr:
                file.write(str(g_a))
                file.write('\n')
        num = num+1

def PaintingHeatmap(cam, img):

    rgb_img = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    return cam_image
