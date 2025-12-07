from torch.utils.data import Dataset
import os.path as osp
from PIL import Image, ImageFile
import functools
import torch


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

class VideoDataset(Dataset):

    def __init__(self, dataset, spatial_transform, temporal_transform, get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return  len(self.dataset)

    def __getitem__(self, index):

        img_paths, pids, camids = self.dataset[index]
        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)   # 从原始的视频序列中采固定帧长的序列

        if isinstance(img_paths,(tuple, list)):
            clips = []
            for i in range(len(img_paths)):
                clip = self.loader(img_paths[i])
                if self.spatial_transform is not None:
                    # self.spatial_transform.randomize_parameters()
                    clip = self.spatial_transform(clip)
                clips.append(torch.stack(clip, 0))
            clips = torch.stack(clips, 0)
        else:
            clip = self.loader(img_paths)

            if self.spatial_transform is not None:
                # self.spatial_transform.randomize_parameters()
                clip = self.spatial_transform(clip)

            # trans T x C x H x W to C x T x H x W
            clips = torch.stack(clip, 0)

        return clips, pids, camids, 0, img_paths