import sys; sys.path.insert(0, '.')
import numpy as np
import sys, os, re, time, shutil, math, random, datetime, argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from TouchWeightsDataset import TouchWeightsDataset
from shared import dataset_tools

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Touch-Weights.')
parser.add_argument('--dataset', default='data/weights/metadata.mat', help="Path to metadata.mat file.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load weights).")
parser.add_argument('--unseenObj', type=int, default=-1, help="Which object to use for test (others are used for train).")
parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help="Just run test and quit.")
parser.add_argument('--snapshotDir', default='.', help="Where to store checkpoints during training.")
args = parser.parse_args()

nFrames = 1
epochs = 200
batch_size = 32
workers = 0

metaFile = args.dataset

class Trainer(object):

    def __init__(self):
        self.init()
        super(Trainer, self).__init__()

    def init(self):
        # Init model
        self.val_loader = self.loadDataset('test', False)

        self.initModel()

    def loadDataset(self, split, shuffle):
        return torch.utils.data.DataLoader(
            TouchWeightsDataset(split=split, doAugment=False, sequenceLength=nFrames, metaFile=metaFile,
            testObjectId=args.unseenObj, testObjectCount=1),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=workers)

    def run(self):
        print('[Trainer] Starting...')

        self.counters = {
            'train': 0,
            'test': 0,
        }

        if args.test:
            self.doSink()
            return

        val_loader = self.val_loader
        train_loader = self.loadDataset('train', True)

        for epoch in range(epochs):
            print('Epoch %d/%d....' % (epoch, epochs))
            self.model.updateLearningRate(epoch)

            self.step(train_loader, self.model.epoch, isTrain = True)
            error,_ = self.step(val_loader, self.model.epoch, isTrain = False)

            # remember best prec@1 and save checkpoint
            is_best = error < self.model.bestPrec
            if is_best:
                self.model.bestPrec = error
            self.model.epoch = epoch + 1

            if not args.snapshotDir is None:
                self.saveCheckpoint(self.model.exportState(), is_best)

        # Final results
        self.doSink()
            
        print('DONE')

    def doSink(self):

        defaultObjects = 'chain, full_can, mug, multimeter, stapler'

        print('Running test for object(s) "%s"...' % (self.val_loader.dataset.meta['objects'][args.unseenObj] if args.unseenObj >= 0 else defaultObjects))
        _, errorTest = self.step(self.val_loader, self.model.epoch)

        print('Test error = %.3f gram.' % errorTest)


    def initModel(self):
        cudnn.benchmark = True

        initShapshot = 'weights'
        if args.unseenObj >= 0:
            initShapshot = '%s_%02d' % (initShapshot, args.unseenObj)
        else:
            initShapshot = '%s_default' % (initShapshot)

        from WeightsModel import WeightsModel as Model

        initShapshot = os.path.join('snapshots', 'weights', initShapshot, 'checkpoint.pth.tar')
        if args.reset:
            initShapshot = None

        self.model = Model(sequenceLength = nFrames)
        self.model.epoch = 0
        self.model.bestPrec = -1e20
        if not initShapshot is None:
            state = torch.load(initShapshot)
            assert not state is None, 'Warning: Could not read checkpoint %s!' % initShapshot
            print('Loading checkpoint %s...' % (initShapshot))
            self.model.importState(state)
            


    def step(self, data_loader, epoch, isTrain = True, sink = False):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = {}

    
        end = time.time()
        for i, (inputs) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputsDict = {
                'image': inputs[1],
                'pressure': inputs[2],
                'objectId': inputs[3],
                'weight': inputs[4],
                }

            res, loss = self.model.step(inputsDict, False, params = {'debug': True})


            for k,v in loss.items():
                if not k in losses:
                    losses[k] = AverageMeter()
                losses[k].update(v, inputs[0].size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if isTrain:
                self.counters['train'] = self.counters['train'] + 1

            msg = '{phase}: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                 epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time,
                phase=('Test'))

            for k,v in losses.items():
                msg += '\t{0} {loss.val:.4f} ({loss.avg:.4f})'.format(k, loss = v)

            print(msg)

        self.counters['test'] = self.counters['test'] + 1

        return losses['Loss'].avg, losses['L1'].avg

    def saveCheckpoint(self, state, is_best):
        snapshotDir = args.snapshotDir
        if not os.path.isdir(snapshotDir):
            os.makedirs(snapshotDir, 0o777)
        chckFile = os.path.join(snapshotDir, 'checkpoint.pth.tar')
        print('Writing checkpoint to %s...' % chckFile)
        torch.save(state, chckFile)
        if is_best:
            bestFile = os.path.join(snapshotDir, 'model_best.pth.tar')
            shutil.copyfile(chckFile, bestFile)
        print('\t...Done.')



    @staticmethod
    def make():        
        random.seed(454878 + time.time() + os.getpid())
        np.random.seed(int(12683 + time.time() + os.getpid()))
        torch.manual_seed(23142 + time.time() + os.getpid())

        ex = Trainer()
        ex.run()



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
if __name__ == "__main__":    
    Trainer.make()

    
    