import torch
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
import torchvision.transforms as T

from Dataset.video.Mars import Mars
from Dataset.video.VideoBase import VideoDataset
from .Sampler import RandomIdentitySampler
import Dataloader.video_transforms as VT
import Dataloader.temporal_transforms as TT

__factory = {
    'mars': Mars,
    # 'msmt17': MSMT17,
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

def make_video_dataloader(cfg):

    """
    根据默认文件的设置，生成训练和测试阶段的迭代器，相当于数据预处理的部分。
    相比图片处理，需要加入空间位置的数据处理和基于时序位置的数据处理。
    """

    # 定义对训练集图片的数据预处理方法（例如：随机反转，随机裁剪等方法）
    # 对视频数据集的处理，需要保证每个序列的帧级特征保持相同处理。

    # 定义对训练集图片在空间维度上数据预处理方法
    train_spatial_transforms = T.Compose([
        VT.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),                                        # 统一尺寸
        VT.RandomHorizontalFlip(p=cfg.INPUT.PROB),                                               # 随机翻转
        VT.Pad(cfg.INPUT.PADDING),                                                               # 填充
        VT.RandomCrop(cfg.INPUT.SIZE_TRAIN),                                                     # 随机裁剪
        VT.ToTensor(),                                                                           # 转换变量名格式
        VT.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),                        # 正则
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),  # 随机擦除
    ])

    # 定义对训练集图片在时间维度上的数据预处理方法
    train_temporal_transforms = TT.TemporalRandomCrop(size=cfg.DATASETS.SEQ_LEN,
                                                      stride=cfg.DATASETS.SAMPLING_STRIDE)      # 这里可以换成我认为更有效的采样方式

    # 定义对测试集图片在时间维度上的数据处理方法
    val_temporal_transforms = TT.DenseCrop(size=cfg.DATASETS.SEQ_LEN)

    # 定义对测试集图片在空间维度上数据预处理方法
    val_spatial_transforms = T.Compose([
        VT.Resize(cfg.INPUT.SIZE_TEST),                                                           # 统一尺寸
        VT.ToTensor(),                                                                            # 转换变量名格式
        VT.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)                          # 正则
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS                                                     # 线程数
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)                          # 定义数据集
    train_set = VideoDataset(dataset.train, train_spatial_transforms, train_temporal_transforms)                                       # 定义训练集
    num_classes = dataset.num_train_pids                                                         # 获取训练集类别总数
    cam_num = dataset.num_train_cams                                                             # 获取训练集相机类别总数（transreid的设置）
    view_num = dataset.num_train_vids                                                            # 获取训练集视角类别总数（transreid的设置）

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

    val_set = VideoDataset(dataset.query + dataset.gallery, val_spatial_transforms, val_temporal_transforms)                        # 测试集
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=num_workers,     # 定义测试集迭代器，为了简便这里将query和gallery按先后顺序合并在一起了。
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num