import numpy as np
import sys, os, re, math, shutil, time, datetime
import scipy.io as sio


def loadMetadata(filename, silent = False):
    '''
    Loads matlab mat file and formats it for simple use.
    '''
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        #metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
        metadata = MatReader().loadmat(filename)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path


class MatReader(object):
    '''
    Loads matlab mat file and formats it for simple use.
    '''

    def __init__(self, flatten1D = True):
        self.flatten1D = flatten1D

    def loadmat(self, filename):
        meta = sio.loadmat(filename, struct_as_record=False) 
        
        meta.pop('__header__', None)
        meta.pop('__version__', None)
        meta.pop('__globals__', None)

        meta = self._squeezeItem(meta)
        return meta

    def _squeezeItem(self, item):
        if isinstance(item, np.ndarray):            
            if item.dtype == np.object:
                if item.size == 1:
                    item = item[0,0]
                else:
                    item = item.squeeze()
            elif item.dtype.type is np.str_:
                item = str(item.squeeze())
            elif self.flatten1D and len(item.shape) == 2 and (item.shape[0] == 1 or item.shape[1] == 1):
                #import pdb; pdb.set_trace()
                item = item.flatten()
            
            if isinstance(item, np.ndarray) and item.dtype == np.object:
                #import pdb; pdb.set_trace()
                #for v in np.nditer(item, flags=['refs_ok'], op_flags=['readwrite']):
                #    v[...] = self._squeezeItem(v)
                it = np.nditer(item, flags=['multi_index','refs_ok'], op_flags=['readwrite'])
                while not it.finished:
                    item[it.multi_index] = self._squeezeItem(item[it.multi_index])
                    it.iternext()



        if isinstance(item, dict):
            for k,v in item.items():
                item[k] = self._squeezeItem(v)
        elif isinstance(item, sio.matlab.mio5_params.mat_struct):
            for k in item._fieldnames:
                v = getattr(item, k)
                setattr(item, k, self._squeezeItem(v))
                 
        return item