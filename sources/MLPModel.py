import torch.nn as nn

from sources.base_model import NegAbs, VOXNOTBaseModel
from sources.params import VOXNOTModelHyperParams

class VOXNOTMLPModel(VOXNOTBaseModel):
    """
    Реализация XNOT для аудио на базе MLP
    """
    def __init__(self, device, hyper_params:VOXNOTModelHyperParams, prod_mode:bool):
        super().__init__(device, hyper_params, prod_mode)

    def _init_model(self, prod_mode):
        # Создание сети T
        self.model_T = nn.Sequential(
            nn.Linear(self.hyper_params.wav_features_size, self.hyper_params.layer_size),
            nn.ReLU(True),
        )

        for l in range(self.hyper_params.layers):
            self.model_T.append(nn.Linear(self.hyper_params.layer_size, self.hyper_params.layer_size))
            self.model_T.append(nn.ReLU(True))

        self.model_T.append(nn.Linear(self.hyper_params.layer_size, self.hyper_params.wav_features_size))

        self.model_T = self.model_T.to(self.device)

        if prod_mode == False:
            # Создание сети F
            self.model_F = nn.Sequential(
                nn.Linear(self.hyper_params.wav_features_size, self.hyper_params.layer_size),
                nn.ReLU(True),
            )

            for l in range(self.hyper_params.layers):
                self.model_F.append(nn.Linear(self.hyper_params.layer_size, self.hyper_params.layer_size))
                self.model_F.append(nn.ReLU(True))

            self.model_F.append(nn.Linear(self.hyper_params.layer_size, self.hyper_params.wav_features_size))
            self.model_F.append(NegAbs())

            self.model_F = self.model_F.to(self.device)

    # Функция расстояния ОТ
    def _cost(self, X, Y):
        return (X - Y).square().flatten(start_dim = 1).mean(dim = 1)

    def _calc_fid(self):
        return abs(self.last_loss_F_value)