import gc
import torch
import os

from sources.audio_helper import VOXNOTFeaturesHelper

# ����� ��� ���������� ������ ��� ����������/���������/������������
# ��� �������� ��������� ���������
# inputDir - ����� � ��������� �����-�������
# outputDir - ����� ���� ����� ������������ �������� � ������
# augmentation_effects - ������� �������� ��� �����������, ������ �� ������� �������� ���� [postfix, [effects]]
# postfix - ��� ��������� � ������������� ����� �����
# ������� �������� ����� �������� ��
# [effects] = [
#   ["lowpass", "-1", "300"], # apply single-pole lowpass filter
#   ["speed", "0.8"],  # reduce the speed
#   ["reverb", "-w"],  # Reverbration gives some dramatic feeling
# ]
# augmentation_count - ������� ��������� �������������� ���������
# keepConvertedAudio - ��������� ����� �� ��������� ������ ���������, ��� �������������� �����
class VOXNOTDatasetPreparationTools:
    OUT_WAV_CHANNEL = 1
    OUT_WAV_ENCODING = 'PCM_S'

    def __init__(self, input_dir:str | os.PathLike, output_dir:str | os.PathLike, augmentation = None, keep_converted_audio:bool = False, device = None, vad_trigger_level = 0):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentation = augmentation
        self.keep_converted_audio = keep_converted_audio
        self.device = device
        self.helper = None
        self.vad_trigger_level = vad_trigger_level
    # ��������� ���������� ������
    def prepare(self):
        if self.helper == None:
            self.helper = VOXNOTFeaturesHelper(self.device)

        for path in os.listdir(self.input_dir):
            source_path = os.path.join(self.input_dir, path)
            if os.path.isfile(source_path):
                self._process_file(source_path)

    def _split_wav(self, wav, wav_rate, interval):
      min_clip_len = interval * wav_rate
      wave_len = wav.shape[1]

      if wave_len < min_clip_len:
        return [wav]
      else:
        return torch.split(wav, min_clip_len, dim = 1)
      return feats

    def _convert_file(self, path):
        """
        ���������� ����� �� ��������� ����� � wav 1 �����(����), 16 ����
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
        ���������� ����� �� ��������� ���������������� ������ � wav 1 �����(����), 16 ����
        """

        return []

    def _process_file(self, path):
        """
        ���������� ����� �� ��������� ������ �����
        ������
        ������������ ���� � wav, 1 �����(����), 16 ����
        ���� ������� �������, �� ���������� �������������� ����� � wav, 1 �����(����), 16 ����
        � ������� WalLM ��������� features
        ��������� features � �����
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


