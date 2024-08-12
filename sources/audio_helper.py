# --------------------------------------------------------
# VOXNOT(other name XNOT-VC): 
# Github source: https://github.com/dmitrii-raketa-erusov/XNOT-VC
# Copyright (c) 2024 Dmitrii Erusov
# Licensed under The MIT License [see LICENSE for details]
# Based on code bases
# https://github.com/pytorch/
# https://github.com/bshall/knn-vc/
# --------------------------------------------------------

import gc
import torch
from torch.functional import Tensor
import torch
import torchaudio

class VOXNOTFeaturesHelper:
    """
    Helper class for extract features from audio and vocoding audio from features
    !!!
    IF NEED YOU MAY DECREASE 
    MAX_DURATION_PER_FILE const
    if you dont have enough of GPU memory. WAlLM use memory proportionality of file duration
    """
    OUT_WAV_RATE = 16000 # REQUIRED BIT RATE FOR WAV FILE 
    MAX_DURATION_PER_FILE = 150 # MAXIMUM DURATION OF WAV FILE
    _knn_vc = None # STATIC, SINGLETON
    def __init__(self, device):
        self.device = device

    def _get_helper(self):
        if (VOXNOTFeaturesHelper._knn_vc == None):
            VOXNOTFeaturesHelper._knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched = True, trust_repo = True, pretrained = True, device = self.device)

        return VOXNOTFeaturesHelper._knn_vc

    def get_features(self, files, vad_trigger_level = 0):
        """
        Extract features from audio with WalLM to TENSOR 
        """
        gc.collect()
        torch.cuda.empty_cache()

        feats = []
        for path in files:
          feats.append(self._get_helper().get_features(path, weights = None, vad_trigger_level = vad_trigger_level))

        feats = torch.concat(feats, dim=0).to(self.device)
        return feats

    def vocode(self, features:Tensor, path):
        """
        Convert features from Tensor to audio with HiFi GAN
        """
        wav_bits = self._get_helper().vocode(features[None].to(self.device)).cpu().squeeze()
        torchaudio.save(path, wav_bits[None], self.OUT_WAV_RATE)

