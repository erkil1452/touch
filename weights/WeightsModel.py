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
from torch.autograd.variable import Variable
import numpy as np


from shared.BaseModel import BaseModel
from shared.resnet_3x3 import resnet18
from TouchWeightsDataset import TouchWeightsDataset

'''
Weight prediction model.
'''

class WeightResnet(nn.Module):
    '''
    Weight regression from a single pressure frame.
    '''

    def __init__(self):
        super(WeightResnet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, 1)

    def forward(self, x):
        x = self.resnet(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class WeightsModel(BaseModel):
    '''
    This class encapsulates the network and handles I/O.
    '''

    @property
    def name(self):
        return 'WeightsModel'


    def initialize(self, baseLr = 1e-3, **kwargs):
        BaseModel.initialize(self)

        self.baseLr = baseLr

        self.model = WeightResnet()
        self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()
        cudnn.benchmark = True

        self.optimizer = torch.optim.Adam([
            {'params': self.model.module.parameters(),'lr_mult': 1.0},          
            ], self.baseLr)

        self.optimizers = [self.optimizer]

        self.criterion = nn.MSELoss().cuda()

        self.epoch = 0
        self.error = 1e20 # last error
        self.bestPrec = 1e20 # best error

        self.dataProcessor = None


            
    def step(self, inputs, isTrain = True, params = {}):

        if isTrain:
            self.model.train()
            assert not inputs['weight'] is None
        else:
            self.model.eval()

        isBackprop = params['isBackprop'] if 'isBackprop' in params else False

        pressure = torch.autograd.Variable(inputs['pressure'].cuda(async=True), requires_grad = (isTrain or isBackprop))
        weight = torch.autograd.Variable(inputs['weight'].cuda(async=True), requires_grad=False) if 'weight' in inputs else None
        if not weight is None:
            weight = self.weightToUniform(weight)
    
        output = self.model(pressure)  
        
        res = {
            'gt': None if weight is None else self.uniformToWeight(weight.data),
            'pred': self.uniformToWeight(output.data),
            'pressure': pressure,
            }

        if weight is None:
            return res, {}

        loss = self.criterion(output, weight)

        if isTrain:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if isTrain:
                self.optimizer.step()
        elif isBackprop:
            output.backward(gradient=weight)

        
        errorL1 = torch.mean(torch.abs(res['pred'] - res['gt']))
        errorL2 = torch.sqrt(torch.mean((res['pred'] - res['gt'])**2))
        losses = OrderedDict([
                            ('Loss', loss.data.item()),
                            ('L1', errorL1.item()),
                            ('L2', errorL2.item()),
                            ])

        return res, losses

    def weightToUniform(self, weight):
        weight = weight / TouchWeightsDataset.WEIGHTS_GAIN
        weight = torch.log(weight + 1.0) / np.log(2.0)
        return weight

    def uniformToWeight(self, weight):
        weight = torch.exp(weight * np.log(2.0)) - 1.0
        weight = weight * TouchWeightsDataset.WEIGHTS_GAIN
        return weight


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