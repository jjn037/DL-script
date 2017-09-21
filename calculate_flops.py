import sys

# sys.path.append('/home/yanglu/workspace/py-RFCN-0425/caffe-rfcn/python')
# sys.path.append('/home/yanglu/workspace/py-RFCN-0425/lib')
sys.path.append('/Users/io/workspace/caffe/python')
sys.path.append('/Users/io/workspace/caffe/lib')
sys.path.append('/Users/io/Documents/opencv-2.4.13/release/lib/')
import caffe
import cv2
import numpy as np

layers_infos = []
layer_info = {}
src_model = '/Users/io/Desktop/deploy_frn66-16x4d.prototxt'
# src_model = './res50.prototxt'
# src_model = '/home/yanglu/workspace/py-faster-rcnn-0302/models/pascal_voc/ResNet101_v2/ResNet101_xie/rcnn_deploy_resnet101_merge_bn_scale.prototxt'
# src_model = '/home/yanglu/merge_bn_scale/resnext/deploy_resnext50_32x4d.prototxt'
img = '/Users/io/Desktop/001.jpeg'
crop_size = (224, 224)
net = caffe.Net(src_model, caffe.TEST)
_img = cv2.imread(img)
_input = cv2.resize(_img, crop_size)
h, w, d = _input.shape
_input = _input.transpose(2, 0, 1)
_input = _input.reshape((1,) + _input.shape)
net.blobs['data'].reshape(*_input.shape)
net.blobs['data'].data[...] = _input
net.blobs['data'].reshape(*_input.shape)
net.forward()
flops = 0
for i, layer in enumerate(net.layers):
    l_name = net._layer_names[i]
    l_bottom = net.bottom_names[l_name]
    # print(l_name)
    if layer.type == 'Convolution':
        # print(net.params[l_name][0].data[:].shape)
        # print(net.blobs[l_name].data[...].shape)
        layer_param = net.params[l_name][0].data[:].shape
        feature_map_size = net.blobs[l_name].data[...].shape
        flops += layer_param[0]*layer_param[1]*layer_param[2]*layer_param[3]* \
                 net.blobs[l_name].data[...].shape[2]*net.blobs[l_name].data[...].shape[3]
print(format(flops,'.2e'))






