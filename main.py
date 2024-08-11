
import torch
from sources.params import VOXNOTModelHyperParams, VOXNOTModelTrainingEnvironment, VOXNOTModelTrainingHyperParams
from sources.VOXNOT import VOXNOT

# Пример использования в работе конвертация речи
def main_vc():
    VOXNOT.clear_mem()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Гипер-параметры модели
    model_hp = VOXNOTModelHyperParams(layers = 4, layer_size = 2048)
    # Конвертация звуковых файлов с речью на обученных моделях
    vx_prod = VOXNOT(device, 'VOXNOTMLPModel', model_hp, True)
    
    # Получение речи 
    vx_prod.make_conversation('Папка или путь к файлу с аудио для конвертации', 'Папка или путь к модели/коэффициентам', 'Папка куда положить результат')

# Пример использования Тренировки моделй
def main_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    VOXNOT.clear_mem()

    # Указываем параметры окружения для тренировки(пути, как логировать)
    environment = VOXNOTModelTrainingEnvironment('путь для хранения лучших моделей', 0, 'путь для сохранения chekpoints', 5000, True, 500)
    
    # Гипер-параметры модели
    model_hp = VOXNOTModelHyperParams(layers = 4, layer_size = 2048)

    vx = VOXNOT(device, 'VOXNOTMLPModel', model_hp, False)
    
    # Тренировки с разными W
    vx.train(True, 'Путь к папке с аудио исходных спикеров', 'Папка с аудио целевого спикера', 'Временная папка для работы', 
             'Папка куда положить лучшую модель', VOXNOTModelTrainingHyperParams(W = 1), environment, "Модель_W1")
    
    # VOXNOTModelTrainingHyperParams(W = 2) - указываем гипер параметр W для тренировки = 2
    vx.train(False, '', ..., VOXNOTModelTrainingHyperParams(W = 2), environment, "Модель_W2")
    vx.train(False, '', ..., VOXNOTModelTrainingHyperParams(W = 4), environment, "Модель_W4")
    vx.train(False, '', ..., VOXNOTModelTrainingHyperParams(W = 8), environment, "Модель_W8")

