import os

import ipdb
import torchvision.transforms as T
from Dataset.image.ImageBase import read_image
from config import cfg
import argparse
import torch
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, EigenGradCAM, FullGrad, EigenCAM
from Models import make_model
import visdom
from Models import make_model
from sklearn.manifold import TSNE



vis = visdom.Visdom(server='http://localhost', port='6006', env='main')

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="configs/MSMT17/ClipBase.yml", help="path to config file", type=str
)

parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args(args=[])

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

source_path = '/root/autodl-tmp/datasets/MSMT17_V1/test'

val_transforms = T.Compose([
    T.Resize([256,128]),                                                           # 统一尺寸
    T.ToTensor(),                                                                  # 转换变量名格式
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                         # 正则
])

file_list = []
for root, dirs, files in os.walk(source_path):
    for file_name in files:
        file_list.append(os.path.join(root, file_name))
random_files = random.sample(file_list, 40)

# fig, axs = plt.subplots(1, 5)
imgs = []
for i, image_path in enumerate(random_files):
    image = cv2.imread(image_path)
    image = image[:, :, ::-1]
    image = cv2.resize(image, (128,256))
    # axs[i].imshow(image)
    # axs[i].axis('off')
    imgs.append(image)

input_tensors = []
rgb_imgs = []
for i, image in enumerate(imgs):
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img)
    input_tensors.append(input_tensor)
    rgb_imgs.append(rgb_img)

# vis.matplot(fig)

model = make_model(cfg, num_classes=1041, camera_num = 15, view_num=0)
model.load_param(cfg.TEST.WEIGHT)
parameters = list(model.classifier.parameters())[0]

para = parameters.detach().numpy()
tsne = TSNE(n_components=2, random_state=42)
embeddings = tsne.fit_transform(para)

plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.show()
vis.matplot(plt)
exit()
def reshape_transform(tensor, height=16, width=8):
    # print(tensor.size())
    tensor = tensor[0].transpose(0,1)
    # print(tensor[0].size())
    result = tensor[:,1:129,:].reshape(1,height, width, 768)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

target_layers = [model.image_encoder.transformer.resblocks[-1].attn]
cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)

fig, axs = plt.subplots(nrows=40, ncols=3, figsize=(8, 200))

for i in range(len(input_tensors)):
    grayscale_cam = cam(input_tensor=input_tensors[i])
    grayscale_cam = grayscale_cam[0, :]

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam_image = show_cam_on_image(rgb_imgs[i], grayscale_cam, use_rgb=True)

    axs[i, 0].imshow(imgs[i])
    axs[i, 0].axis('off')
    axs[i, 1].imshow(heatmap)
    axs[i, 1].axis('off')
    axs[i, 2].imshow(cam_image)
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()
vis.matplot(fig)