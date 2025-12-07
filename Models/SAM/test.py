import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



# image = cv2.imread('/root/autodl-tmp/datasets/MSMT17_V1/train/0359/0359_007_07_0113morning_0934_0.jpg')
image = cv2.imread('/root/code/SAM/notebooks/images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image1 = cv2.resize(image, (256, 128), interpolation=cv2.INTER_LINEAR)
# input_image_torch1 = torch.as_tensor(image1).cuda()
# input_image_torch1 = input_image_torch1.permute(2, 0, 1).contiguous()[None, :, :, :]
# print(image.shape)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()

import sys
sys.path.append("../..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
#
sam_checkpoint = "/root/code/SME/sam_vit_b_01ec64.pth"
model_type = "vit_b"
#
device = "cuda"
#
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
#
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
#
print(len(masks))
print(masks[0].keys())
print(masks[0]['area'])
exit()
#
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

## 我想验证一下如果按照我们常规的数据处理，分割的效果会怎么样？

# image1 = cv2.resize(image, (256, 128), interpolation=cv2.INTER_LINEAR)
# input_image_torch1 = torch.as_tensor(image1).cuda()
# input_image_torch1 = input_image_torch1.permute(2, 0, 1).contiguous()[None, :, :, :]
# mask_generator.generate(input_image_torch1, transforms=False)


