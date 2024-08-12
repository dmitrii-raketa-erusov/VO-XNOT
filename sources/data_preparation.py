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
import os
import torchaudio

from sources.audio_helper import VOXNOTFeaturesHelper

class VOXNOTDatasetPreparationTools:
    """
    Класс для подготовки датасетов для тренировки/валидации/тестирования
    
    Берет входные audio любых форматов, преобразовывает их в файлы для извлечения audio-features(wav, 16k битрейт, моно, длительностью до 150 сек.).
    В случае, если исходный файл более чем 150 сек, нарезает его на куски по 150 сек. максимум
    Далее, из этих wav файлов извлекаются audio-features с помощью WalLM, а получаемые Tensors сериализуются в файлы, которые и являются dataset'ами для 
    алгоритма обучения
    """
    OUT_WAV_CHANNEL = 1 # Сколько каналов в выходном файле
    OUT_WAV_ENCODING = 'PCM_S' # Формат wav

    def __init__(self, input_dir:str | os.PathLike, output_dir:str | os.PathLike, augmentation = None, keep_converted_audio:bool = False, device = None, vad_trigger_level = 0):
        """
        inputDir - папка с исходными аудио-файлами
        outputDir - папка куда будут записываться датасеты с audio-features
        augmentation - матрица эффектов для аугментации, список из матрицы эффектов виде [postfix, [effects]]
        postfix - что добавлять к оригинальному имени файла
        матрица эффектов может состоять из
        [effects] = [
          ["lowpass", "-1", "300"], # apply single-pole lowpass filter
          ["speed", "0.8"],  # reduce the speed
          ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]
        augmentation_count - сколько случайных преобразований выполнять
        keepConvertedAudio - указывает нужно ли сохранять, полученные из исходного аудио, промежуточные аудио файлы, из которых вытягивались audio-features 
        vad_trigger_level - уровень обрезки по громкости, по умолчанию 0, звуки любой громкости 
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentation = augmentation
        self.keep_converted_audio = keep_converted_audio
        self.device = device
        self.helper = None
        self.vad_trigger_level = vad_trigger_level
    
    def prepare(self):
        """
        Запускает подготовку dataset'ов
        """
        if self.helper == None:
            self.helper = VOXNOTFeaturesHelper(self.device)

        for path in os.listdir(self.input_dir):
            source_path = os.path.join(self.input_dir, path)
            if os.path.isfile(source_path):
                self._process_file(source_path)

    def _split_wav(self, wav, wav_rate, interval):
      """
      Метод нарезки wav на несколько файлов фиксированной максимальной длины
      """
      min_clip_len = interval * wav_rate
      wave_len = wav.shape[1]

      if wave_len < min_clip_len:
        return [wav]
      else:
        return torch.split(wav, min_clip_len, dim = 1)
      return feats

    def _convert_file(self, path):
        """
        Внутренний метод по обработке исходного файла любого формата, в wav 1 канал(моно), 16 кбит
        """
        path_converted = os.path.join(self.output_dir, os.path.basename(path)) + '.wav'
        waveform, sample_rate = torchaudio.load(path)
        transform = torchaudio.transforms.Resample(sample_rate, VOXNOTFeaturesHelper.OUT_WAV_RATE)
        waveform_sampled = transform(waveform)
        channel_0 = (waveform_sampled[0])[None]

        wavs = self._split_wav(channel_0, VOXNOTFeaturesHelper.OUT_WAV_RATE, VOXNOTFeaturesHelper.MAX_DURATION_PER_FILE)

        out_files = []

        for index, wav in enumerate(wavs):
            path_converted = f'{os.path.join(self.output_dir, os.path.basename(path))}_sl{index}.wav'
            out_files.append(path_converted)
            torchaudio.save(path_converted, wav, VOXNOTFeaturesHelper.OUT_WAV_RATE, encoding = self.OUT_WAV_ENCODING, format = "wav")

        del waveform
        del waveform_sampled
        del channel_0
        del wavs

        return out_files

    def _generate_augm_files(self, path):
        """
        Внутренний метод по генерации аугментированных файлов в wav 1 канал(моно), 16 кбит
        в гит не выложен, тестируется...
        """

        return []

    def _process_file(self, path):
        """
        Внутренний метод по обработке одного файла
        
        конвертирует файл в wav, 1 канал(моно), 16 кбит
        если указаны эффекты аугментации, то генерирует дополнительные файлы в wav, 1 канал(моно), 16 кбит в случайном порядке
        с помощью WalLM извлекает audio-features
        сериализует Tensor с audifeatures в файл
        """
        gc.collect()
        torch.cuda.empty_cache()

        _files_to_process = self._convert_file(path)
        #_files_to_process.append(self._generate_augm_files(_files_to_process[0])

        features = self.helper.get_features(_files_to_process, self.vad_trigger_level)
        torch.save(features, _files_to_process[0] + ".pt")
        del features

        if self.keep_converted_audio == False:
            for file_to_remove in _files_to_process:
                os.remove(file_to_remove)

        return True


