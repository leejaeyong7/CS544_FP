import os
from os import path

import torch
from torch import optim

class CheckPoint:
    '''
    Handles loading / saving model / optimizer / scheduler / epoch and
    other variables used for running network.
    '''
    @staticmethod
    def save(checkpoint_path, model, 
             optimizer=None, scheduler=None, total_step=None, epoch=None):
        if(not path.exists(path.dirname(checkpoint_path))):
            os.makedirs(path.dirname(checkpoint_path))
        # save model (of course)
        checkpoint = {
            'model':None,
            'optimizer':None,
            'scheduler':None,
            'epoch':None,
            'total_step':None,
        }
        checkpoint['model'] = model.state_dict()
        # if(optimizer):
        #     checkpoint['optimizer'] = optimizer.state_dict()
        # if(scheduler):
        #     checkpoint['scheduler'] = scheduler
        # if(epoch):
        #     checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def load(checkpoint_path, device=None):
        if(not path.exists(checkpoint_path)):
            raise Exception("Checkpoint Path Not Found!")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint

