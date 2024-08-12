# --------------------------------------------------------
# VOXNOT(other name XNOT-VC): 
# Github source: https://github.com/dmitrii-raketa-erusov/XNOT-VC
# Copyright (c) 2024 Dmitrii Erusov
# Licensed under The MIT License [see LICENSE for details]
# Based on code bases
# https://github.com/pytorch/
# --------------------------------------------------------

import torch
from sources.params import VOXNOTModelHyperParams, VOXNOTModelTrainingEnvironment, VOXNOTModelTrainingHyperParams
from sources.VOXNOT import VOXNOT

# Пример использования в работе конвертация речи
# Конвертация звуковых файлов с речью на обученных моделях
def main_vc():
    # Чистим память
    VOXNOT.clear_mem()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Гипер-параметры модели
    model_hp = VOXNOTModelHyperParams(layers = 4, layer_size = 1024)
    # Используем простую модель MLP, 
    vx_prod = VOXNOT(device, 'VOXNOTMLPModel', model_hp, True)
    
    # Конверсия речи из файлов в папке /content/query 
    # в целевую речь по обученной модели, файлы с целевой речь положить в папку /content/out
    vx_prod.make_conversation('/content/query', '/models/речь_лягушки.pt', '/content/out')

    # Конверсия речи из файла /content/new query.mp3 
    # в целевую речь по обученной модели, целевой файл /content/out_file.wav
    vx_prod.make_conversation('/content/query', '/models/мужик_из_ютуб.pt', '/content/out_file.wav')

# Пример использования Тренировки моделей
def main_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Чистим память
    VOXNOT.clear_mem()

    # Указываем параметры окружения для тренировки(пути, как логировать)
    environment = VOXNOTModelTrainingEnvironment('путь для хранения лучших моделей', 0, 'путь для сохранения chekpoints', 5000, True, 500)
    
    # Гипер-параметры модели
    model_hp = VOXNOTModelHyperParams(layers = 4, layer_size = 1024)
    
    # Создаем модель в режиме тренировки, параметр prod_mode = False
    vx = VOXNOT(device, 'VOXNOTMLPModel', model_hp, False)
    
    # Тренировки с разными W и разными целевыми спикерами на конвертацию своего голоса в голоса целевых спикеров
    
    # W = 8, целевой спикер "мужик с ютуба"
    vx.train(True, '/content/my_voice.mp3', '/content/мужик.m4a', '/content/tmp', 
             '/content/models', VOXNOTModelTrainingHyperParams(W = 8), environment, "мужик с ютуба_W8")
    
    # W = 12, целевой спикер "мужик с ютуба", передаем FALSE, так как заново готовить dataset не нужно
    vx.train(False, '/content/my_voice.mp3', '/content/мужик.m4a', '/content/tmp', 
             '/content/models', VOXNOTModelTrainingHyperParams(W = 12), environment, "мужик с ютуба_W12")

    # W = 12, целевой спикер "крокодил Гена", передаем 1ый параметр TRUE, так как нужен новый dataset, входной файл целевого спикера другой
    vx.train(True, '/content/my_voice.mp3', '/content/gena_samples/', '/content/tmp', 
             'Папка куда положить лучшую модель', VOXNOTModelTrainingHyperParams(W = 12), environment, "крокодил Гена")


