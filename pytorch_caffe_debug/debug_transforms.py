import os
import sys
# sys.path.append('/Users/io/Documents/opencv-2.4.13/release/lib/')
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cv2
from PIL import Image
import os.path

mean_value = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
images_pytorch = []
images_pil = []


rootdir = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/debug/val_temp'

# for parent,dirnames,filenames in os.walk(rootdir + '/val'):
#     for filename in filenames:
#         img_path = os.path.join(parent, filename)
#         print img_path
def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('/home/yanglu/Database/ILSVRC2017/Data/CLS-LOC/val', transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    for parent,dirnames,filenames in os.walk('/home/yanglu/Database/ILSVRC2017/Data/CLS-LOC/val'):
        for filename in filenames:
            img_path = os.path.join(parent,filename)
            print img_path
            _img = Image.open(img_path).convert('RGB')
            # print(_img.mode)
            _img = scale(_img)
            _img = center_crop(_img)
            _img = image_preprocess(_img)
            _img = np.asarray([_img], dtype=np.float32)
            _img = _img.transpose(0, 3, 1, 2)
            images_pytorch.append(_img)
    i = 0
    j = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        print inputs.numpy().all() == images_pytorch[batch_idx].all()
        if not inputs.numpy().all() == images_pytorch[batch_idx].all():
            j += 1

        i += 1
    print j
    print i


def image_preprocess(img):
    r, g, b = img.split()
    r, g, b = np.array(r).astype(np.float32), np.array(g).astype(np.float32), np.array(b).astype(np.float32)
    r, g, b = np.divide(r, 255.).astype(np.float32), np.divide(g, 255.).astype(np.float32), np.divide(b, 255.).astype(np.float32)
    r, g, b = np.divide(r-mean_value[0], std[0]).astype(np.float32), np.divide(g-mean_value[1], std[1]).astype(np.float32), np.divide(b-mean_value[2], std[2]).astype(np.float32)
    _img = cv2.merge([r, g, b])
    return _img


def center_crop(img): # single crop
    # print img.shape
    w, h = img.size
    th, tw = 224, 224
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))


def scale(img):
    w, h = img.size
    if (w <= h and w == 256) or (h <= w and h == 256):
        return img
    if w < h:
        ow = 256
        oh = int(256 * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = 256
        ow = int(256 * w / h)
        return img.resize((ow, oh), Image.BILINEAR)

if __name__ == '__main__':
    main()
