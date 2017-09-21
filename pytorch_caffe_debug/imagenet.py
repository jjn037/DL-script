'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function
import sys
sys.path.append('/home/yanglu/workspace/py-faster-rcnn-0302/caffe-fast-rcnn/python')
import caffe
import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import res

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig



model_weights = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/rex26/deploy_resnext26_32x4d.caffemodel'
model_deploy = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/rex26/deploy_resnext26_32x4d.prototxt'
# model_weights = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/resnet18/deploy-res18-0703.caffemodel'
# model_deploy = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/resnet18/deploy-res18-0629.prototxt'
# model_weights = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/irb/deploy_irb101-0711.caffemodel'
# model_deploy = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/irb/deploy_irb101-0709.prototxt'


prob_layer = 'prob'
gpu_mode = True
gpu_id = 0

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
net = caffe.Net(model_deploy, model_weights, caffe.TEST)
top_k = (1, 5)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext26',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
caffe_accuracy = np.zeros(len(top_k))
pytorch_accuracy = np.zeros(len(top_k))

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    args.data = '/home/yanglu/merge_bn_scale/jjn/pytorch2caffe-resx/debug'
    traindir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    args.pretrained = True
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = res.__dict__[args.arch]()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    args.evaluate = True
    if args.evaluate:
        print('\nEvaluation only')
        _input = test(val_loader, model, criterion, start_epoch, use_cuda)
        # print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return _input
    #
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(val_loader))
    i = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        # outputs = model(inputs)
        # # print(type(outputs))
        # pytorch_index = (-[outputs.data.cpu().numpy()][0]).argsort()[0]
        # # print(pytorch_index.shape)
        # loss = criterion(outputs, targets)
        # print('Pytorch Testing image: ' + str(batch_idx + 1) + '/' + str(50000) + '  ' + str(pytorch_index[0]) + '/' + str(
        #     targets.data.cpu().numpy()), )
        # for j in xrange(len(top_k)):
        #     # print(type(targets))
        #     if targets.data.cpu().numpy() in pytorch_index[:top_k[j]]:
        #         pytorch_accuracy[j] += 1
        #     tmp_acc = float(pytorch_accuracy[j]) / float(batch_idx + 1)
        #     if top_k[j] == 1:
        #         print('\tpytorch_top_' + str(top_k[j]) + ':' + str(tmp_acc), )
        #     else:
        #         print('pytorch_top_' + str(top_k[j]) + ':' + str(tmp_acc))
        # print(batch_idx)
        # # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))
        # print('pytorch_top1: ' + str(top1.avg))
        # print('pytorch_top5: ' + str(top5.avg))
        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        _input = inputs.data.cpu().numpy()
        _input = _input.copy()
        net.blobs['data'].reshape(*_input.shape)
        net.blobs['data'].data[...] = _input
        net.forward()
        # print(np.sum(net.blobs[prob_layer].data, axis=0))
        outputs_caffe = net.blobs['prob'].data
        score_index = ((-((outputs_caffe)[0])).argsort())
        print('Caffe Testing image: ' + str(batch_idx + 1) + '/' + str(50000) + '  ' + str(score_index[0]) + '/' + str(
            targets.data.cpu().numpy()), )
        for j in range(len(top_k)):
            # print(type(targets))
            if targets.data.cpu().numpy() in score_index[:top_k[j]]:
                caffe_accuracy[j] += 1
            tmp_acc = float(caffe_accuracy[j]) / float(batch_idx + 1)
            if top_k[j] == 1:
                print('\tcaffe_top_' + str(top_k[j]) + ':' + str(tmp_acc), )
            else:
                print('caffe_top_' + str(top_k[j]) + ':' + str(tmp_acc))

    return


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    inputs = main()