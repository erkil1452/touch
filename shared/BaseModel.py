import os
import torch

class BaseModel(object):
    
    def __init__(self, *args, **kwargs):
        self.initialize(*args, **kwargs)

    @property
    def name(self):
        return 'BaseModel'

    @property
    def inputSize(self):
        return (0, 0)

    def initialize(self):
        pass

            
    def step(self, inputs, isTrain = True, params = {}):
        pass
    
    def vizualizeResults(self, results, index, size = (512, 512)):
        pass

    def importState(self, state):
        pass

    def exportState(self):
        return {}

    def createDataProcessor(self):
        return None

    def updateLearningRate(self, epoch):
        pass

    def setParams(self, **kwargs):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate_schedulers(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('Updated learning rate = %.7f' % lr)

    def update_learning_rate_manual(self, epoch, baseLr = 0.001, period = 30):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = baseLr * (0.1 ** (epoch // period))
        print('\tNew lr = %.5f' % lr)
        for i,optimizer in enumerate(self.optimizers):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr