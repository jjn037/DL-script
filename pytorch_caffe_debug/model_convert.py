import sys

sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/caffe-fast-rcnn/python')

import caffe
import torch
import numpy as np
'''
convert align_inception_resnet101
'''
pytorch_weight = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/irb/model_best.pth.tar'
caffe_weight = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/irb/deploy_irb101-0711.caffemodel'
caffe_deploy = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/irb/deploy_irb101-0709.prototxt'

caffe.set_mode_cpu()
net = caffe.Net(caffe_deploy, caffe.TEST)
_model = torch.load(pytorch_weight)
model = _model['state_dict']
pytorch_layers = model.keys()
pytorch_layers = []
pytorch_match_layers = []
pytorch_conv_layers = []
pytorch_bn_layers = []
for layer in model.keys():
    if 'downsample' in layer:
        pytorch_match_layers.append(layer)
    elif 'bn' in layer:
        pytorch_bn_layers.append(layer)
    else:
        pytorch_conv_layers.append(layer)
caffe_layers = []
caffe_downsample_layers = []
caffe_conv_layers = []
caffe_bn_layers = []
for i, layer in enumerate(net._layer_names):
    if net.layers[i].type not in ['Convolution', "BatchNorm", "Scale", 'InnerProduct']:
        continue
    if 'match' in layer:
        caffe_downsample_layers.append(layer)
    elif net.layers[i].type in ["BatchNorm", "Scale"]:
        caffe_bn_layers.append(layer)
    else:
        caffe_conv_layers.append(layer)

for i in range((len(caffe_bn_layers)) / 2):
    print caffe_bn_layers[i*2], pytorch_bn_layers[i * 4 + 2]
    net.params[caffe_bn_layers[i*2]][0].data[...] = model[pytorch_bn_layers[i * 4 + 2]].cpu().numpy()
    print caffe_bn_layers[i*2], pytorch_bn_layers[i * 4 + 3]
    net.params[caffe_bn_layers[i*2]][1].data[...] = model[pytorch_bn_layers[i * 4 + 3]].cpu().numpy()
    net.params[caffe_bn_layers[i*2]][2].data[...] = 1.0
    print caffe_bn_layers[i*2+1], pytorch_bn_layers[i * 4 + 0]
    net.params[caffe_bn_layers[i*2+1]][0].data[...] = model[pytorch_bn_layers[i * 4 + 0]].cpu().numpy()
    print caffe_bn_layers[i*2+1], pytorch_bn_layers[i * 4 + 1]
    net.params[caffe_bn_layers[i*2+1]][1].data[...] = model[pytorch_bn_layers[i * 4 + 1]].cpu().numpy()

for i in range((len(caffe_conv_layers))):
    net.params[caffe_conv_layers[i]][0].data[...] = model[pytorch_conv_layers[i]].cpu().numpy()

for i in range((len(caffe_downsample_layers)) / 3):
    print caffe_downsample_layers[i * 3], pytorch_match_layers[i * 5]
    net.params[caffe_downsample_layers[i * 3]][0].data[...] = model[pytorch_match_layers[i * 5]].cpu().numpy()
    print caffe_downsample_layers[i * 3 + 1], pytorch_match_layers[i * 5 + 3]
    net.params[caffe_downsample_layers[i * 3 + 1]][0].data[...] = model[pytorch_match_layers[i * 5 + 3]].cpu().numpy()
    print caffe_downsample_layers[i * 3 + 1], pytorch_match_layers[i * 5 + 4]
    net.params[caffe_downsample_layers[i * 3 + 1]][1].data[...] = model[pytorch_match_layers[i * 5 + 4]].cpu().numpy()
    net.params[caffe_downsample_layers[i * 3 + 1]][2].data[...] = 1.0
    print caffe_downsample_layers[i * 3 + 2], pytorch_match_layers[i * 5 + 1]
    net.params[caffe_downsample_layers[i * 3 + 2]][0].data[...] = model[pytorch_match_layers[i * 5 + 1]].cpu().numpy()
    print caffe_downsample_layers[i * 3 + 2], pytorch_match_layers[i * 5 + 2]
    net.params[caffe_downsample_layers[i * 3 + 2]][1].data[...] = model[pytorch_match_layers[i * 5 + 2]].cpu().numpy()
net.params[caffe_conv_layers[-1]][1].data[...] = model[pytorch_conv_layers[-1]].cpu().numpy()
net.save(caffe_weight)
