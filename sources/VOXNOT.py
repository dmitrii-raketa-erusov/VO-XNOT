import gc
import os
import torch

from sources.params import VOXNOTModelHyperParams, VOXNOTModelTrainingEnvironment, VOXNOTModelTrainingHyperParams
from sources.base_model import VOXNOTBaseModel
from sources.data_preparation import VOXNOTDatasetPreparationTools
from sources.voxnot_dataset import VOXNOTDataset
from sources.audio_helper import VOXNOTFeaturesHelper

import shutil

class VOXNOT:
    """
    Класс для работы с алгоритмом. Тренировки моделей, вычисление результатов
Метод вычисления(файл, модель, выходной файл)
Метод для работы по папкам(папка с файлами, папка с моделями, выходная папка)
Нацеливаем папки
- папка с входным аудиофайлом/файлами
- папка моделей
Делает несколько конвертаций в выходной вав по загруженным моделям
"""
    model_instance:VOXNOTBaseModel

    def _clear_folder(dir):
      if os.path.exists(dir):
          for the_file in os.listdir(dir):
              file_path = os.path.join(dir, the_file)

              try:
                  if os.path.isfile(file_path):
                      os.remove(file_path)
                  else:
                      VOXNOT._clear_folder(file_path)
                      os.rmdir(file_path)

                  print(f'removed {file_path}')
              except Exception as e:
                  print(e)

    def __init__(self, device, model_class_name:str, hyper_params:VOXNOTModelHyperParams, prod_mode:bool):
        self.device = device
        class_object = globals()[model_class_name]
        self.model_instance = class_object(device, hyper_params, prod_mode)

    def clear_mem():
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
          torch.cuda.empty_cache()

    def _prepare_dataset(self, delete_last_prepared_data, input_dir, dataset_dir):
        exists_prepared_datasets = False

        for file_ds in os.listdir(dataset_dir):
          path_file_ds = os.path.join(dataset_dir, file_ds)
          if os.path.isfile(path_file_ds) and os.path.splitext(path_file_ds)[1] == '.pt':
            exists_prepared_datasets = True
            break

        if delete_last_prepared_data == True or exists_prepared_datasets == False:
          print(f'Preparing datasets {input_dir}..')

          if os.path.isdir(dataset_dir) == True:
              VOXNOT._clear_folder(dataset_dir)
          else:
            os.mkdir(dataset_dir)

          tool = VOXNOTDatasetPreparationTools(input_dir, dataset_dir, augmentation = None, keep_converted_audio = True, device = self.device, vad_trigger_level=0)
          tool.prepare()

        VOXNOT.clear_mem()

        datasets = []

#        dataset = VOXNOTDataset([])
#        for file_ds in glob(os.path.join(dataset_dir, '*.pt')):
#            dataset.concat(VOXNOTDataset(file_ds))
#        return dataset

        for file_ds in os.listdir(dataset_dir):
          path_file_ds = os.path.join(dataset_dir, file_ds)
          if os.path.isfile(path_file_ds) and os.path.splitext(path_file_ds)[1] == '.pt':
            datasets += [VOXNOTDataset(path_file_ds, self.device)]

        return ConcatDataset(datasets)


    def train(self, delete_last_prepared_data:bool, input_query_dir:str | os.PathLike, input_reffer_dir:str | os.PathLike,
              temp_dir:str | os.PathLike, output_dir:str | os.PathLike,
              training_hyper_params:VOXNOTModelTrainingHyperParams, training_env:VOXNOTModelTrainingEnvironment,
              training_name:str):
        """
        Метод тренировки модели
        delete_last_prepared_data - удалить предыдущие тренировочные датасеты и сделать новые
        input_query_dir - папка с аудио исходных спикеров
        input_reffer_dir - папка с аудио целевых спикеров
        temp_dir - временная директория для работы
        output_dir - папка для выходных моделей
        training_hyper_params - гиперпараметры тренировки
        training_env - параметры окружения 
        training_name - название модели(будет в выходном файле)
        """

        dataset_X = self._prepare_dataset(delete_last_prepared_data, input_query_dir, os.path.join(temp_dir, "input_ds_X"))
        dataset_Y = self._prepare_dataset(delete_last_prepared_data, input_reffer_dir, os.path.join(temp_dir, "input_ds_Y"))

        self.model_instance.set_train_params(training_hyper_params, training_env, dataset_X, dataset_Y, training_name)
        self.model_instance.train()

        model_path = self.model_instance.get_last_best_model_path()
        model_dest_path = os.path.join(output_dir, f'{training_name}.pt')

        print(f'Copy model {model_path} to {model_dest_path}')
        shutil.copy2(model_path, model_dest_path)

    def _get_files(self, path:str | os.PathLike):
        file_list = []

        if os.path.isdir(path) == True:
            for file_path in os.listdir(path):
              if os.path.isfile(os.path.join(path, file_path)) == True:
                file_list.append(os.path.join(path, file_path))
        else:
            file_list.append(path)

        return file_list


    def make_conversation(self, query_path:str | os.PathLike, trained_model_path:str | os.PathLike, out_path:str | os.PathLike):
        """
        Метод конвертации речи
        query_path - Папка или путь к файлу с аудио для конвертации
        trained_model_path - Папка или путь к модели/коэффициентам, которые были получены после тренировки
        out_path - Папка куда положить результат, wav файлы с результатом
        """        
        query_file_list = self._get_files(query_path)
        model_file_list = self._get_files(trained_model_path)

        helper = VOXNOTFeaturesHelper(self.device)

        for model in model_file_list:
            self.model_instance.load_weights(model)
            model_name = os.path.basename(model)

            for query_path in query_file_list:
                X = helper.get_features([query_path, ])
                Y = self.model_instance.predict(X)

                out_file_path = out_path if os.path.isdir(out_path) == False else os.path.join(out_path, f"{os.path.basename(query_path)}_{model_name}.wav")

                helper.vocode(Y, out_file_path)

