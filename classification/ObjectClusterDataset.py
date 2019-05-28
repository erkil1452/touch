import os, random, math
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import re, itertools, sys
from scipy import misc as scm

import sklearn
import sklearn.cluster
import sklearn.decomposition

from ObjectDataset import ObjectDataset


class ObjectClusterDataset(ObjectDataset):
    ''' Chooses N frames from clusters (made by k-mean clustering) in TSNE projection space. '''

    def __init__(self, useClusters = True, **kwargs):
        self.useClusters = useClusters
        self.clusters = None
        super(ObjectClusterDataset, self).__init__(**kwargs)
        self.refresh()
    
    def getName(self):
        return 'ObjectClusterDataset'

    
    def initClusters(self):
        print('[ObjectClusterDataset] Initializing class clusters...')

        allIndices = np.argwhere(self.validN).flatten()
        recordingIds = self.meta['recordingId'][allIndices]
        recordings = np.unique(recordingIds)

        nEmbDims = 8     
        self.clusters = {}  
        self.clusterIds = np.zeros((self.pressure.shape[0],), int)

        # Every recording
        for i,recordingId in enumerate(recordings):
            rMask = recordingIds == recordingId
            rIndices = allIndices[rMask]

            objectIds = self.meta['objectId'][rIndices]
            objects = np.unique(objectIds)

            # Every object (though there is always just one per recording)
            recClusters = {}
            for j,objectId in enumerate(objects):
                print('\tClustering %s / %s (#%d)...' % (self.meta['recordings'][recordingId], self.meta['objects'][objectId], objectId))
                oMask = objectIds == objectId
                oIndices = rIndices[oMask]
                samples = self.pressure[oIndices,...]
                samples = samples.reshape(samples.shape[0],-1)

                # embedding
                pca = sklearn.decomposition.PCA(n_components=nEmbDims)
                Xemb = pca.fit_transform(samples)

                x_min = np.min(Xemb)
                x_max = np.max(Xemb)
                Xemb = (Xemb - x_min) / (x_max - x_min)
                
                kmeans = sklearn.cluster.KMeans(n_clusters=self.sequenceLength, init='k-means++', n_init=50).fit(Xemb)
                
                recClusters[objectId] = []
                for clusterId in range(self.sequenceLength):
                    cMask = kmeans.labels_ == clusterId
                    cIndices = oIndices[cMask]
                    recClusters[objectId] += [cIndices]
                    self.clusterIds[cIndices] = clusterId


            self.clusters[recordingId] = recClusters


    def refresh(self):
        # generates new tuples
        if not self.useClusters:
            ObjectDataset.refresh(self)
            return

        if self.clusters is None:
            self.initClusters()

        print('[ObjectClusterDataset] Refreshing tuples...')
        indices0 = np.argwhere(self.valid0).flatten()
        self.indices = np.zeros((len(indices0), self.sequenceLength), int)
        for i in range(len(indices0)):
            ind0 = indices0[i]

            recordingId = self.meta['recordingId'][ind0]
            objectId = self.meta['objectId'][ind0]
            clusterId = self.clusterIds[ind0]

            # select from each cluster
            others = []
            for j in range(0, self.sequenceLength - 1):
                otherClusterId = j
                if otherClusterId >= clusterId:
                    otherClusterId += 1
                
                candidates = self.clusters[recordingId][objectId][otherClusterId]
                others += [candidates[np.random.randint(0,len(candidates))]]

            # shuffle order
            np.random.shuffle(others)
            self.indices[i,0] = ind0
            self.indices[i,1:] = others



   
        
