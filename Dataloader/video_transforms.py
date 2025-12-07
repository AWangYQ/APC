import torchvision.transforms as T
import random
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


"""
    自带的代码只能处理单帧的图片，这里让其能同时处理一个序列的图片。
    这里继承这些类，然后在输出的时候做一些处理。
    返回的都是list对象。
"""

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, imgs):
        filp_imgs = []
        if random.random() < self.p:
            for img in imgs:
                filp_imgs.append(F.hflip(img))
            return filp_imgs
        else:
            return imgs

class Resize(T.Resize):

    def __init__(self, size=[128, 256], interpolation=InterpolationMode.BILINEAR):
        super(Resize, self).__init__(size, interpolation)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        resize_imgs = []
        for img in imgs:
            resize_imgs.append(F.resize(img, self.size, self.interpolation))
        return resize_imgs

class Pad(T.Pad):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        super(Pad, self).__init__(padding, fill, padding_mode)

    def __call__(self, imgs):
        pad_imgs = []
        for img in imgs:
            pad_imgs.append(F.pad(img, self.padding, self.fill, self.padding_mode))

        return pad_imgs

class RandomCrop(T.RandomCrop):
    def __init__(self, size, padding=0, pad_if_needed=False):
        super(RandomCrop, self).__init__(size, padding, pad_if_needed)

    def __call__(self, imgs):
        # print(len(imgs))
        i, j, h, w = self.get_params(imgs[0], self.size)

        crop_imgs = []
        for img in imgs:
            crop_imgs.append(F.crop(img, i, j, h, w))

        return crop_imgs

class ToTensor(T.ToTensor):

    def __call__(self, imgs):
        tensor_imgs = []
        for img in imgs:
            tensor_imgs.append(F.to_tensor(img))
        return tensor_imgs

class Normalize(T.Normalize):
    def __init__(self, mean, std):
        super(Normalize, self).__init__(mean, std)

    def __call__(self, imgs):
        nor_imgs = []
        nor_imgs = []
        for img in imgs:
            nor_imgs.append(F.normalize(img, self.mean, self.std))
        return nor_imgs