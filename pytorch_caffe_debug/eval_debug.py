import sys

sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/caffe-fast-rcnn/python')

import numpy as np
import caffe
import cv2
import datetime
import imagenet
import res
import torch
import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
import os.path

'''
debug
caffe evaluate script data
equal input for caffe and pytorch model

'''

gpu_mode = True
gpu_id = 0
data_root = '/home/yanglu/Database/ILSVRC2017'
val_file = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/debug/ILSVRC2012_val1000.txt'
save_log = 'log{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
model_weights = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/resnet18/deploy-res18-0703.caffemodel'
model_deploy = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/resnet18/deploy-res18-0629.prototxt'
prob_layer = 'prob'
class_num = 1000
base_size = 256  # short size
crop_size = 224
# mean_value = np.array([128.0, 128.0, 128.0])  # BGR
mean_value = np.array([0.485, 0.456, 0.406])  # RGB
# mean_value = np.array([103.94000244, 116.77999878, 123.68000031])
# std = np.array([128.0, 128.0, 128.0])  # BGR
std = np.array([0.229, 0.224, 0.225])  # RGB
# std = np.array([57.375, 57.12, 58.395])  # BGR
crop_num = 1  # 1 and others for center(single)-crop, 12 for mirror(12)-crop, 144 for multi(144)-crop
batch_size = 1
top_k = (1, 5)
caffe_right_results = []
pytorch_right_results = []
caffe_not_in_pytorch = []
pytorch_not_in_caffe = []
different_results = []

model = res.__dict__['resnet18'](pretrained=True)

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)


def eval_batch():
    eval_images = []
    ground_truth = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip().split(' ')[0])
        ground_truth.append(int(i.strip().split(' ')[1]))
    f.close()

    skip_num = 0
    eval_len = len(eval_images)
    caffe_accuracy = np.zeros(len(top_k))
    pytorch_accuracy = np.zeros(len(top_k))
    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        _img = cv2.imread(data_root + eval_images[i + skip_num])
        _img = cv2.resize(_img, (int(_img.shape[1] * base_size / min(_img.shape[:2])),
                                 int(_img.shape[0] * base_size / min(_img.shape[:2])))
                          )
        _img = image_preprocess(_img)
        caffe_score_vec = np.zeros(class_num, dtype=np.float32)
        pytorch_score_vec = np.zeros(class_num, dtype=np.float32)
        crops = []
        if crop_num == 1:
            crops.append(center_crop(_img))
        elif crop_num == 12:
            crops.extend(mirror_crop(_img))
        elif crop_num == 144:
            crops.extend(multi_crop(_img))
        else:
            crops.append(center_crop(_img))

        iter_num = int(len(crops) / batch_size)
        for j in xrange(iter_num):
            pytorch_score_vec += pytorch_process(np.asarray(crops, dtype=np.float32)[j * batch_size:(j + 1) * batch_size])
            caffe_score_vec += caffe_process(np.asarray(crops, dtype=np.float32)[j * batch_size:(j + 1) * batch_size])
        #
        caffe_score_index = (-caffe_score_vec / len(crops)).argsort()
        pytorch_score_index = (-pytorch_score_vec/len(crops)).argsort()
        # pytorch_score_index = (-pytorch_score_vec / len(crops)).argsort()
        print caffe_score_index
        print 'okkkk'
        print pytorch_score_index
        if caffe_score_index[0] != caffe_score_index[0]:
            different_results.append(eval_images[i + skip_num])
            print eval_images[i + skip_num]
            # print caffe_score_index[0], caffe_score_index[0], ground_truth[i + skip_num]

        print 'Caffe Testing image: ' + str(i + 1) + '/' + str(eval_len - skip_num) + '  ' + str(
            caffe_score_index[0]) + '/' + str(
            ground_truth[i + skip_num]),

        for j in xrange(len(top_k)):
            print caffe_score_index[:top_k[j]]
            if ground_truth[i + skip_num] in caffe_score_index[:top_k[j]]:
                caffe_accuracy[j] += 1
                # if top_k[j] ==1 :
                #     print caffe_right_results.append(eval_images[i + skip_num])
            tmp_acc = float(caffe_accuracy[j]) / float(i + 1)
            if top_k[j] == 1:
                print '\ttop_' + str(top_k[j]) + ':' + str(tmp_acc),
            else:
                print 'top_' + str(top_k[j]) + ':' + str(tmp_acc)
        # print pytorch_score_index
        print 'Pytorch Testing image: ' + str(i + 1) + '/' + str(eval_len - skip_num) + '  ' + str(
            pytorch_score_index[0]) + '/' + str(
            ground_truth[i + skip_num]),
        for j in xrange(len(top_k)):
            # print pytorch_score_index[:top_k[j]]
            if ground_truth[i + skip_num] in pytorch_score_index[:top_k[j]]:
                pytorch_accuracy[j] += 1
                # if top_k[j] ==1 :
                #     print pytorch_right_results.append(eval_images[i + skip_num])
            tmp_acc = float(pytorch_accuracy[j]) / float(i + 1)
            if top_k[j] == 1:
                print '\ttop_' + str(top_k[j]) + ':' + str(tmp_acc),
            else:
                print 'top_' + str(top_k[j]) + ':' + str(tmp_acc)
    print different_results
    end_time = datetime.datetime.now()
    w = open(save_log, 'w')
    s1 = 'Evaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    s2 = '\nThe model is: {}. \nThe val file is: {}. \n{} images has been tested, crop_num is: {}, base_size is: {}, ' \
         'crop_size is: {}.'.format(model_weights, val_file, str(eval_len), str(crop_num), str(base_size),
                                    str(crop_size))
    s3 = '\nThe mean value is: ({}, {}, {}).'.format(str(mean_value[0]), str(mean_value[1]), str(mean_value[2]))
    s4 = ''
    for i in xrange(len(top_k)):
        _acc = float(accuracy[i]) / float(eval_len)
        s4 += '\nAccuracy of top_{} is: {}; correct num is {}.'.format(str(top_k[i]), str(_acc), str(int(accuracy[i])))
    print s1, s2, s3, s4
    # w.write(s1 + s2 + s3 + s4)
    w.close()


def image_preprocess(img):
    b, g, r = cv2.split(img)
    b, g, r = b / 255.0, g / 255.0, r / 255.0
    b, g, r = (b - mean_value[2]) / std[2], (g - mean_value[1]) / std[1], (r - mean_value[0]) / std[0]
    _img = cv2.merge([r, g, b])
    return _img


def center_crop(img):  # single crop
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]


def over_sample(img):  # 12 crops of image
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    sample_list = [img[:crop_size, :crop_size], img[-crop_size:, -crop_size:], img[:crop_size, -crop_size:],
                   img[-crop_size:, :crop_size], img[yy: yy + crop_size, xx: xx + crop_size],
                   cv2.resize(img, (crop_size, crop_size))]
    return sample_list


def mirror_crop(img):  # 12*len(size_list) crops
    crop_list = []
    img_resize = cv2.resize(img, (base_size, base_size))
    mirror = img_resize[:, ::-1]
    crop_list.extend(over_sample(img_resize))
    crop_list.extend(over_sample(mirror))
    return crop_list


def multi_crop(img):  # 144(12*12) crops
    crop_list = []
    size_list = [256, 288, 320, 352]  # crop_size: 224
    # size_list = [270, 300, 330, 360]  # crop_size: 235
    # size_list = [320, 352, 384, 416]  # crop_size: 299
    # size_list = [352, 384, 416, 448]  # crop_size: 320
    short_edge = min(img.shape[:2])
    for i in size_list:
        img_resize = cv2.resize(img, (img.shape[1] * i / short_edge, img.shape[0] * i / short_edge))
        yy = int((img_resize.shape[0] - i) / 2)
        xx = int((img_resize.shape[1] - i) / 2)
        for j in xrange(3):
            left_center_right = img_resize[yy * j: yy * j + i, xx * j: xx * j + i]
            mirror = left_center_right[:, ::-1]
            crop_list.extend(over_sample(left_center_right))
            crop_list.extend(over_sample(mirror))
    return crop_list


def caffe_process(_input):
    _input = _input.copy()
    _input = _input.transpose(0, 3, 1, 2)
    net.blobs['data'].reshape(*_input.shape)
    net.blobs['data'].data[...] = _input
    net.forward()
    return np.sum(net.blobs['prob'].data, axis=0)


def dataloader:
    rootdir = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/debug/val_temp'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(rootdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)


def pytorch_process(_input):
    _input = _input.copy()
    _input = _input.transpose(0, 3, 1, 2)
    inputs = torch.from_numpy(_input)
    inputs = torch.autograd.Variable(inputs, volatile=True)
    outputs = model(inputs)
    outputs = outputs.data.numpy()
    return np.sum(outputs, axis=0)


if __name__ == '__main__':
    eval_batch()


