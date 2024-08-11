import gc
import torch
from torch.functional import Tensor
import torch
import torchaudio

class VOXNOTFeaturesHelper:
    """
    Features helper for extract features from audio and vocoding audio from features
    """
    OUT_WAV_RATE = 16000
    MAX_DURATION_PER_FILE = 150
    _knn_vc = None
    def __init__(self, device):
        self.device = device

    def _get_helper(self):
        if (VOXNOTFeaturesHelper._knn_vc == None):
            VOXNOTFeaturesHelper._knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched = True, trust_repo = True, pretrained = True, device = self.device)

        return VOXNOTFeaturesHelper._knn_vc

    def get_features(self, files, vad_trigger_level = 0):
        """
        extract features from audio with WalLM 
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
        Convert features to audio with HiFi GAN
        """
        wav_bits = self._get_helper().vocode(features[None].to(self.device)).cpu().squeeze()
        torchaudio.save(path, wav_bits[None], self.OUT_WAV_RATE)

