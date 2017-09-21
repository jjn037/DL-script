"""
data from preprocess by myself for caffe and pytorch
"""
import numpy as np
# import caffe
import cv2
import datetime
import random
import torch
from PIL import Image
import res

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

# print("=> using pre-trained model '{}'".format('resnet18'))
# model = res.__dict__['resnet18'](pretrained=True)
model = res.__dict__['align_inception_resnet101']()

gpu_mode = True
gpu_id = 0
data_root = '/home/yanglu/Database/ILSVRC2017'
# val_file = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/debug/ILSVRC2012_val1000.txt'
val_file = '/home/yanglu/merge_bn_scale/ILSVRC2012_val.txt'
save_log = 'log{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

class_num = 1000
base_size = 256 # short size
crop_size = 224
# mean_value = np.array([128.0, 128.0, 128.0])  # BGR
mean_value = np.array([0.485, 0.456, 0.406])  # RGB
# mean_value = np.array([103.94000244, 116.77999878, 123.68000031])
# std = np.array([128.0, 128.0, 128.0])  # BGR
std = np.array([0.229, 0.224, 0.225])   #RGB
# std = np.array([57.375, 57.12, 58.395])  # BGR
crop_num = 1  # 1 and others for center(single)-crop, 12 for mirror(12)-crop, 144 for multi(144)-crop
batch_size = 1
top_k = (1, 5)
use_cuda = torch.cuda.is_available()

# Random seed
manualSeed = None
if manualSeed is None:
    manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)


def eval_batch():

    # switch to evaluate mode
    model.eval()

    eval_images = []
    ground_truth = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip().split(' ')[0])
        ground_truth.append(int(i.strip().split(' ')[1]))
    f.close()

    skip_num = 0
    eval_len = len(eval_images)
    accuracy = np.zeros(len(top_k))
    caffe_accuracy = np.zeros(len(top_k))
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = Image.open(data_root + eval_images[i + skip_num]).convert('RGB')
        # print(_img.mode)
        _img = scale(_img)
        _img = center_crop(_img)
        _img = image_preprocess(_img)
        _img = np.asarray([_img], dtype=np.float32)
        _img = _img.transpose(0, 3, 1, 2)

        ############################pytorch###########################
        pytorch_img = torch.from_numpy(_img.copy())
        inputs = pytorch_img.cuda()
        inputs = torch.autograd.Variable(inputs, volatile=True)
        # model = model
        # compute output
        outputs = model(inputs).cuda()
        # print(type(outputs))
        score_index = (-[outputs.data.cpu().numpy()][0]).argsort()[0]
        # print(pytorch_index.shape)
        print 'Testing image: ' + str(i + 1) + '/' + str(eval_len - skip_num) + '  ' + str(score_index[0]) + '/' + str(
            ground_truth[i + skip_num]),
        pytorch_cls_correct = True
        if score_index[0] != ground_truth[i + skip_num]:
            pytorch_wrong_images.append(eval_images[i + skip_num])
            pytorch_cls_correct = False
        for j in xrange(len(top_k)):
            print score_index[:top_k[j]]
            if ground_truth[i + skip_num] in score_index[:top_k[j]]:
                accuracy[j] += 1
            tmp_acc = float(accuracy[j]) / float(i + 1)
            if top_k[j] == 1:
                print '\ttop_' + str(top_k[j]) + ':' + str(tmp_acc),
            # else:
            #     print 'top_' + str(top_k[j]) + ':' + str(tmp_acc)


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
    eval_batch()


