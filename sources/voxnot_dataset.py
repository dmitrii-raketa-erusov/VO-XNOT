# --------------------------------------------------------
# VOXNOT(other name XNOT-VC): 
# Github source: https://github.com/dmitrii-raketa-erusov/XNOT-VC
# Copyright (c) 2024 Dmitrii Erusov
# Licensed under The MIT License [see LICENSE for details]
# Based on code bases
# https://github.com/pytorch/
# --------------------------------------------------------

import gc
import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn
import os
from torch.utils.data.dataset import random_split
from torch.utils.data import  DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F

class Sampler:
    """
    Base class for Sample
    """
    def __init__(
        self, device,
    ):
        self.device = device

    def sample(self, size = 5):
        pass

class LoaderSampler(Sampler):
    """
    Loader class, as in PyTorch example
    """
    def __init__(self, loader, device):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size = 5):
        assert size <= self.loader.batch_size
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch[:size].to(self.device)

class VOXNOTDataset:
    """
    Dataset for audio-features
    """
    tensor: torch.Tensor
    def __init__(self, tensor: str | torch.Tensor | os.PathLike, device):
        """
        tensor - Path to serialized Tensor with data or Tensor with audio-features
        """
        if (type(tensor) in [torch.Tensor]):
            self.tensor = tensor
        else:
            self.tensor = torch.load(tensor, map_location=torch.device(device))

    def shape(self):
        return self.tensor.shape

    def concat(self, dataset):
        """
        Concat 2 datasets with audio-feature in one, if you dont want to use ConcatDataset
        """
        self.tensor = torch.concat([self.tensor, dataset.tensor], dim = 0).to(self.device)

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        return (self.tensor[index])

