import os, random, math
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
import re, itertools
from scipy import misc as scm

from shared import dataset_tools

class MetadataLoader(object):

    instance = None

    def __init__(self):        
        self.metas = dict()

    def getMeta(self, metaFile):
        if not metaFile in self.metas:
            self.metas[metaFile] = dataset_tools.loadMetadata(metaFile)
        return self.metas[metaFile]
        
    @staticmethod
    def getInstance():
        if MetadataLoader.instance is None:
            MetadataLoader.instance = MetadataLoader()
        return MetadataLoader.instance

class TouchDataset(data.Dataset):

    def initialize(self, split = 'train', doAugment = False, sequenceLength = 5, inputSize = 32):


        self.doAugment = doAugment
        self.sequenceLength = sequenceLength
        self.indices = None
        self.inputSize = inputSize
        

        self.dummyImage = torch.zeros((3, 1, 1))

        self.meta, self.valid0, self.validN, self.pressure = self.loadDataset()


        if split == 'test':
            self.split = 'test'
        elif split == 'train':
            self.split = 'train'
        elif split == 'all':
            self.split = 'all'
        else:
            raise RuntimeError('Unknown split "%s"!' % split)

        if not self.split == 'all':
            splitId = np.argwhere(self.meta['splits'] == self.split).flatten()[0]
            self.valid0 = np.logical_and(self.valid0, self.meta['splitId'].flatten() == splitId)
            self.validN = np.logical_and(self.validN, self.meta['splitId'].flatten() == splitId)

        # Generate tuples
        np.random.seed(157843)
        self.refresh()

        print('Loaded "%s" - split "%s" with %d records...' % (self.getName(), self.split, len(self.indices)))

    def refresh(self):
        # Generate tuples
        print('[TouchDataset] Refreshing tuples...')
        if not self.indices is None and self.sequenceLength <= 1:
            return

        indices0 = np.argwhere(self.valid0).flatten()
        self.indices = np.zeros((len(indices0), self.sequenceLength), int)
        for i in range(len(indices0)):
            ind0 = indices0[i]
            mask = self.validN.copy()
            if 'recordingId' in self.meta:
                mask = np.logical_and(mask, self.meta['recordingId'] == self.meta['recordingId'][ind0])
            mask[ind0] = False
            candidates = np.argwhere(mask).flatten()
            np.random.shuffle(candidates)
            self.indices[i,0] = ind0
            self.indices[i,1:] = candidates[:(self.sequenceLength - 1)]


    def getName(self):
        return None

    def loadDataset(self):
        return {}, [], [], [] # meta, valid0, validN, pressure

    def transformPressure(self, pressure):
        pressure = np.clip((pressure.astype(np.float32) - 500) / (650 - 500), 0.0, 1.0)
        return pressure
        
  

    def getItemBase(self, indexExt):
        inds = self.indices[indexExt,:]
        index = inds[0]
        batch = self.meta['batches'][self.meta['batchId'][index]]
        recording = self.meta['recordings'][self.meta['recordingId'][index]]
        frame = self.meta['frame'][index]

        # Pressure
        pressure = self.pressure[inds,...]
        pressure = torch.from_numpy(pressure)

        if self.doAugment:
            noise = torch.randn_like(pressure) * 0.015
            pressure += noise

        # image
        image = self.dummyImage
            
        # to tensor
        row = torch.LongTensor([int(index)])
        
        return row, image, pressure


    
        
    def __len__(self):
        return len(self.indices)


def clamp(x, a, b):
    return np.minimum(np.maximum(x, a), b)
