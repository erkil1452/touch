import os, random, math
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import re, itertools

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

class TouchWeightsDataset(data.Dataset):
    
    WEIGHTS_GAIN = 800

    def __init__(self, split = 'train', doAugment = False, sequenceLength = 1, metaFile = 'metadata_2018-06-17.mat',
            testObjectId=-1, testObjectCount=1):

        print('Loading Touch weights dataset...')

        self.doAugment = doAugment
        self.sequenceLength = sequenceLength
        self.testObjectId = testObjectId
        self.testObjectCount = testObjectCount

        self.loadDataset(metaFile)
        if split == 'test':
            self.split = 'test'
        elif split == 'train':
            self.split = 'train'
        elif split == 'all':
            self.split = 'all'
        else:
            raise RuntimeError('Unknown split "%s"!' % split)

        if self.testObjectId >= 0:
            # use a custom split instead of the default one in the metadata.mat
            objectIds = np.arange(len(self.meta['objects']) - 1)
            print('\tIn total has %d objects.' % (len(objectIds)))

            isTest = self.meta['objectId'].flatten() == self.testObjectId
            print('\tSelected #%d (%s) as a primary test object.' % (self.testObjectId, self.meta['objects'][self.testObjectId]))                

            np.random.seed(451239 + self.testObjectId)
            np.random.shuffle(objectIds)
            for i in range(1,self.testObjectCount):
                oId = objectIds[i - 1]
                if oId >= self.testObjectId:
                    oId += 1
                isTest = np.logical_or(isTest, self.meta['objectId'].flatten() == oId)
                print('\tSelected #%d (%s) as a secondary test object #%d.' % (oId, self.meta['objects'][oId], i))

            # Define new split
            self.meta['splitId'][isTest] = np.argwhere(self.meta['splits'] == 'test').flatten()[0]
            self.meta['splitId'][np.logical_not(isTest)] = np.argwhere(self.meta['splits'] == 'train').flatten()[0]


        if not self.split == 'all':
            splitId = np.argwhere(self.meta['splits'] == self.split).flatten()[0]
            self.valid0 = np.logical_and(self.valid0, self.meta['splitId'].flatten() == splitId)
            self.validN = np.logical_and(self.validN, self.meta['splitId'].flatten() == splitId)

        # Generate tuples
        np.random.seed(157843)
        indices0 = np.argwhere(self.valid0).flatten()
        self.indices = np.zeros((len(indices0), self.sequenceLength), int)
        for i in range(len(indices0)):
            ind0 = indices0[i]
            mask = np.logical_and(self.validN, self.meta['recordingId'] == self.meta['recordingId'][ind0])
            mask = np.logical_and(mask, self.meta['graspIndex'] == self.meta['graspIndex'][ind0])
            mask[ind0] = False
            candidates = np.argwhere(mask).flatten()
            np.random.shuffle(candidates)
            self.indices[i,0] = ind0
            self.indices[i,1:] = candidates[:(self.sequenceLength - 1)]

        print('Loaded Touch weights dataset split "%s" with %d records...' % (self.split, len(self.indices)))


    def loadDataset(self, metaFile):
        self.meta = MetadataLoader.getInstance().getMeta(metaFile)
        valid = self.meta['graspValid'].flatten().astype(np.bool)
        self.validN = valid # for samples 1+
        self.valid0 = self.validN
        
        pressure = self.meta['pressure']
        pressure = np.clip((pressure.astype(np.float32) - 500) / (650 - 500), 0.0, 1.0)
        self.pressure = pressure

        print('>> Using %d frames out of %d in meta => %.2f%%' % (np.sum(self.valid0), len(self.valid0), np.sum(self.valid0) / len(self.valid0) * 100))

        

  

    def __getitem__(self, indexExt):
        inds = self.indices[indexExt,:]
        index = inds[0]
        batch = self.meta['batches'][self.meta['batchId'][index]]
        recording = self.meta['recordings'][self.meta['recordingId'][index]]
        frame = self.meta['frame'][index]
        objectId = self.meta['objectId'][index]

        # Pressure
        pressure = self.pressure[inds,...]
        pressure = torch.from_numpy(pressure)

        weight = self.meta['weights'][objectId]

        # image is just dummy, not used
        image = pressure[:3,...]
            
        # to tensor
        row = torch.LongTensor([int(index)])
        objectId = torch.LongTensor([int(objectId)])
        weight = torch.FloatTensor([weight])

        
        return row, image, pressure, objectId, weight

    
        
    def __len__(self):
        return len(self.indices)


def clamp(x, a, b):
    return np.minimum(np.maximum(x, a), b)