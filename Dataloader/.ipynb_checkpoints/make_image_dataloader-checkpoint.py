import torch
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
import numpy as np

from Dataset.image.Market1501 import Market1501
from Dataset.image.MSMT17 import MSMT17
from Dataset.image.ImageBase import ImageDataset
from .Sampler import RandomIdentitySampler
import ipdb
from Dataset.image.ImageBase import read_image
import traceback
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
}

def train_collate_fn(batch):

    """
    collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果。
    如果不设置，这返回 ImageDataset 类的返回结果。 这里相当于重新组织了一下在训练和测试迭代器中的输出结果。
    """

    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def val_collate_fn(batch):

    """
    道理和 train_collater_fn 是类似的，不同的地方是本方法是针对测试阶段做的设置。
    """

    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_image_dataloader(cfg):

    """
    根据默认文件的设置，生成训练和测试阶段的迭代器，相当于数据预处理的部分。
    """

    # 定义对训练集图片的数据预处理方法（例如：随机反转，随机裁剪等方法）
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),                                        # 统一尺寸
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),                                               # 随机翻转
        T.Pad(cfg.INPUT.PADDING),                                                               # 填充
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),                                                     # 随机裁剪
        T.ToTensor(),                                                                           # 转换变量名格式
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),                        # 正则
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),  # 随机擦除
    ])

    #定义对测试集图片的数据预处理方法
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),                                                           # 统一尺寸
        T.ToTensor(),                                                                            # 转换变量名格式
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)                          # 正则
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS                                                     # 线程数
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)                          # 定义数据集
    train_set = ImageDataset(dataset.train, train_transforms)                                    # 定义训练集
    num_classes = dataset.num_train_pids                                                         # 获取训练集类别总数
    cam_num = dataset.num_train_cams                                                             # 获取训练集相机类别总数（transreid的设置）
    view_num = dataset.num_train_vids                                                            # 获取训练集视角类别总数（transreid的设置）

    # train_query = QueryDataset(dataset.id_set, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        """
        PxK 采样，生成训练的迭代器
        """
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

    query_set = ImageDataset(dataset.query, val_transforms)                        # 测试集
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    query_loader = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,     # 定义测试集迭代器，为了简便这里将query和gallery按先后顺序合并在一起了。
        collate_fn=val_collate_fn
    )
    gallery_loader = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,     
        collate_fn=val_collate_fn
    )
    
    train_query_set = []
    for _ in range(11*120):
        train_query_set.append(data_prepare(dataset.id_set, train_transforms))

    return train_loader, query_loader, gallery_loader, len(dataset.query), num_classes, cam_num, view_num, train_query_set


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.data[index]
        img = read_image(img_path)
        img = self.transform(img)
        return img, pid

def data_prepare(dict, transform):
    samples = []
    for i in range(len(dict)):
        selected_sample = random.choice(dict[i])
        samples.append(selected_sample)

    imgs = []
    pids = []
    
    val_set = ImageData(samples, transform)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=8)

    return val_loader