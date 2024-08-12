# --------------------------------------------------------
# VOXNOT(other name XNOT-VC): 
# Github source: https://github.com/dmitrii-raketa-erusov/XNOT-VC
# Copyright (c) 2024 Dmitrii Erusov
# Licensed under The MIT License [see LICENSE for details]
# Based on code bases
# https://github.com/pytorch/
# --------------------------------------------------------

import torch.nn as nn

from sources.base_model import NegAbs, VOXNOTBaseModel
from sources.params import VOXNOTModelHyperParams

class VOXNOTMLPModel(VOXNOTBaseModel):
    """
    Реализация XNOT для аудио-конверсии на базе MLP 
    """
    def __init__(self, device, hyper_params:VOXNOTModelHyperParams, prod_mode:bool):
        super().__init__(device, hyper_params, prod_mode)

    def _init_model(self, prod_mode):
        """
        Переопределяем метод базового класса, для создания Т и F, которые используются в XNOT алгоритме(как описано в оригинальной статье)
        в данном случае и T, и F - это MLP. Чего достаточно, так как пространство audio-features латентное и простое
        """

        self.model_T = nn.Sequential(
            nn.Linear(self.hyper_params.wav_features_size, self.hyper_params.layer_size),
            nn.ReLU(True),
        )

        # создаем столько слоев, сколько скрытых слоев в параметре layer_size
        for l in range(self.hyper_params.layers):
            self.model_T.append(nn.Linear(self.hyper_params.layer_size, self.hyper_params.layer_size))
            self.model_T.append(nn.ReLU(True))

        self.model_T.append(nn.Linear(self.hyper_params.layer_size, self.hyper_params.wav_features_size))

        self.model_T = self.model_T.to(self.device)

        # если указано, что не production режим, то создаем F для тренировки
        if prod_mode == False:
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

    # Переопределяем 
    # Функция стоимости расстояния из алгоритма ОТ
    # можно подставлять и другие, например стандартную mse из PyTorch, но так нагляднее
    def _cost(self, X, Y):
        return (X - Y).square().flatten(start_dim = 1).mean(dim = 1)

    # Переопределяем метод расчета fad(в случае audio нужно считать fad)
    # пока же для тестов возвращаем просто значение loss функции потенциала F
    # для нас чем оно меньше, тем лучше результат
    # в будующей реализации определим fad, как описано в PyTorch
    def _calc_fid(self):
        return abs(self.last_loss_F_value)