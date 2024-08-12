# --------------------------------------------------------
# VOXNOT(other name XNOT-VC): 
# Github source: https://github.com/dmitrii-raketa-erusov/XNOT-VC
# Copyright (c) 2024 Dmitrii Erusov
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os


class VOXNOTModelHyperParams:
    """
    Hyper parameters for VOXNOT model
    layers - count of linear-layers and ReLU. Default = 2
    layer_size - width of each lienear layer. Default = 1024
    wav_features_size - size of each vector with audio features
    """
    def __init__(self,
                    layers:int = 2, layer_size:int = 1024, wav_features_size:int = 1024
                    ):
        self.layers = layers
        self.wav_features_size = wav_features_size
        self.layer_size = layer_size

class VOXNOTModelTrainingHyperParams:
    """
    Hyper parameters for training VOXNOT model
    W - main parameter of equality. Default value is 8
    iters - cycle for T, default = 10
    max_steps - max count of epochs
    test_proportion - proportion of test data in datasets        
    """
    def __init__(self,
                 W:int = 8, iters:int = 10, max_steps:int = 5000,
                 LR:float = 1e-4, weight_decay:float = 1e-10,
                 test_proportion:float = 0.01, batch_size:int = None
                 ):
        if batch_size == None:
            batch_size = 8 if W < 9 else 4

        self.iters = iters
        self.max_steps = max_steps
        self.W = W

        self.LR = LR
        self.weight_decay = weight_decay

        self.test_proportion = test_proportion
        self.batch_size = 8 * self.W * 1

class VOXNOTModelTrainingEnvironment:
    """
    Hyper parameters for training environment
    best_point_path - path to folder for storing model with best result
    check_point_path - path to folder for checkpoints
    check_point_interval - interval for save checkpoint
    write_loss_interval - set to interval for logging loss-value. 0 - every time
    check_fid_interval - check FID interval
    overwrite_cp_files - True if need to rewrite checkpoints files
    """
    def __init__(self,
                 best_point_path:str | os.PathLike, check_fid_interval:int,
                 check_point_path:str | os.PathLike, check_point_interval:int, overwrite_cp_files:bool,
                 write_loss_interval:int):
        self.best_point_path = best_point_path

        self.check_point_path = check_point_path
        self.check_point_interval = check_point_interval
        self.overwrite_cp_files = overwrite_cp_files

        self.check_fid_interval = check_fid_interval

        self.write_loss_interval = write_loss_interval


