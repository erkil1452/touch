import os, random, math
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
import re, itertools

from TouchDataset import TouchDataset, MetadataLoader


class ObjectDataset(TouchDataset):
    def __init__(self, split='train', doAugment = False, doFilter = True, doBalance=True, sequenceLength = 5, objectId = None, notObjectId = None,
        metaFile = '', inputSize = 32):

        self.metaFile = metaFile
        self.doFilter = doFilter
        self.doBalance = doBalance
        self.objectId = objectId
        self.notObjectId = notObjectId
        
        TouchDataset.initialize(self, split=split, doAugment=doAugment, sequenceLength=sequenceLength, inputSize = inputSize)

    
    def getName(self):
        return 'ObjectDataset'

    def loadDataset(self):
        meta = MetadataLoader.getInstance().getMeta(self.metaFile)
        valid = np.ones((len(meta['frame']),), np.bool)
        if not self.objectId is None:
            isCorrectObject = meta['objectId'].flatten() == self.objectId
            valid = np.logical_and(valid, isCorrectObject)
        if not self.notObjectId is None:
            isCorrectObject = meta['objectId'].flatten() == self.notObjectId
            valid = np.logical_and(valid, np.logical_not(isCorrectObject))
        if self.doFilter:
            hasValidLabel = meta['hasValidLabel'].flatten().astype(np.bool)
            valid = np.logical_and(valid, hasValidLabel)
        validN = valid # for samples 1+

        if self.doBalance:
            isBalanced = meta['isBalanced'].flatten().astype(np.bool)
            valid0 = np.logical_and(validN, isBalanced) # for sample 0
        else:
            valid0 = validN
        
        pressure = self.transformPressure(meta['pressure'])

        return meta, valid0, validN, pressure

    

            
    



    def __getitem__(self, indexExt):
        row, image, pressure = self.getItemBase(indexExt)
        index = row[0]
        objectId = self.meta['objectId'][index]

        # to tensor
        objectId = torch.LongTensor([int(objectId)])

        #import pdb; pdb.set_trace()
        
        return row, image, pressure, objectId

        
