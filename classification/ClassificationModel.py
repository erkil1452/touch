import argparse
import os
import shutil
import time, math, datetime, re
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.autograd.variable import Variable

from shared.BaseModel import BaseModel
from shared.resnet_3x3 import resnet18
from shared import dataset_tools


''' The final model '''

class SmallNet(nn.Module):
    '''
    This model extract features for each single input frame.
    '''

    def __init__(self):
        super(SmallNet, self).__init__()
        self.features = resnet18(pretrained=False)
        self.features.fc  = nn.Threshold(-1e20, -1e20) # a pass-through layer for snapshot compatibility


    def forward(self, pressure):
        x = self.features(pressure)
        return x

       
class TouchNet(nn.Module):
    '''
    This model represents our classification network for 1..N input frames.
    '''

    def __init__(self, num_classes=1000, nFrames=5):
        super(TouchNet, self).__init__()
        self.net = SmallNet()
        self.combination = nn.Conv2d(128*nFrames, 128, kernel_size=1, padding=0)
        self.classifier = nn.Linear(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        xs = []
        # CNN of each input frame
        for i in range(x.size(1)):
            xi = x[:,i:i+1,...]
            xi = self.net(xi)
            xs += [xi]
        x = torch.cat(xs, dim=1)

        # combine
        x = self.combination(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x



class ClassificationModel(BaseModel):
    '''
    This class encapsulates the network and handles I/O.
    '''

    @property
    def name(self):
        return 'ClassificationModel'


    def initialize(self, numClasses, sequenceLength = 1, baseLr = 1e-3):
        BaseModel.initialize(self)

        print('Base LR = %e' % baseLr)
        self.baseLr = baseLr
        self.numClasses = numClasses
        self.sequenceLength = sequenceLength

        self.model = TouchNet(num_classes = self.numClasses, nFrames = self.sequenceLength)
        self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        cudnn.benchmark = True

        self.optimizer = torch.optim.Adam([
            {'params': self.model.module.parameters(),'lr_mult': 1.0},          
            ], self.baseLr)

        self.optimizers = [self.optimizer]

        self.criterion = nn.CrossEntropyLoss().cuda()

        self.epoch = 0
        self.error = 1e20 # last error
        self.bestPrec = 1e20 # best error

        self.dataProcessor = None

            
    def step(self, inputs, isTrain = True, params = {}):

        if isTrain:
            self.model.train()
            assert not inputs['objectId'] is None
        else:
            self.model.eval()


        image = torch.autograd.Variable(inputs['image'].cuda(async=True), requires_grad = (isTrain))
        pressure = torch.autograd.Variable(inputs['pressure'].cuda(async=True), requires_grad = (isTrain))
        objectId = torch.autograd.Variable(inputs['objectId'].cuda(async=True), requires_grad=False) if 'objectId' in inputs else None
    
        if isTrain:
            output = self.model(pressure)   
        else:
            with torch.no_grad():
                output = self.model(pressure)  
        
        _, pred = output.data.topk(1, 1, True, True)
        res = {
            'gt': None if objectId is None else objectId.data,
            'pred': pred,
            }

        if objectId is None:
            return res, {}

        loss = self.criterion(output, objectId.view(-1))

        (prec1, prec3) = self.accuracy(output, objectId, topk=(1, min(3, self.numClasses)))

        if isTrain:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        losses = OrderedDict([
                            ('Loss', loss.data.item()),
                            ('Top1', prec1),
                            ('Top3', prec3),
                            ])

        return res, losses

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.data.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0], res[1]


    def importState(self, save):
        params = save['state_dict']
        if hasattr(self.model, 'module'):
            try:
                self.model.load_state_dict(params, strict=True)
            except:
                self.model.module.load_state_dict(params, strict=True)
        else:
            params = self._clearState(params)
            self.model.load_state_dict(params, strict=True)

        self.epoch = save['epoch'] if 'epoch' in save else 0
        self.bestPrec = save['best_prec1'] if 'best_prec1' in save else 1e20
        self.error = save['error'] if 'error' in save else 1e20
        print('Imported checkpoint for epoch %05d with loss = %.3f...' % (self.epoch, self.bestPrec))


    def _clearState(self, params):
        res = dict()
        for k,v in params.items():
            kNew = re.sub('^module\.', '', k)
            res[kNew] = v

        return res
        

    def exportState(self):
        dt = datetime.datetime.now()
        state = self.model.state_dict()
        for k in state.keys():
            #state[k] = state[k].share_memory_()
            state[k] = state[k].cpu()
        return {
            'state_dict': state,
            'epoch': self.epoch,
            'error': self.error,
            'best_prec1': self.bestPrec,
            'datetime': dt.strftime("%Y-%m-%d %H:%M:%S")
            }


    def updateLearningRate(self, epoch):
        self.adjust_learning_rate_new(epoch, self.baseLr)


    def adjust_learning_rate_new(self, epoch, base_lr, period = 100): # train for 2x100 epochs
        gamma = 0.1 ** (1.0/period)
        lr_default = base_lr * (gamma ** (epoch))
        print('New lr_default = %f' % lr_default)

        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr_mult'] * lr_default