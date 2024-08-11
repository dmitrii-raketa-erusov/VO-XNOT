import gc
import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn
import os
from torch.utils.data.dataset import random_split
from torch.utils.data import  DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F

from params import VOXNOTModelTrainingHyperParams, VOXNOTModelHyperParams, VOXNOTModelTrainingEnvironment
from voxnot_dataset import LoaderSampler

class NegAbs(nn.Module):
    def __init__(self):
        super(NegAbs, self).__init__()

    def forward(self, input):
        return -torch.abs(input)
    

class VOXNOTBaseModel:
    """
    Базовая класс реализующий алгоритм XNOT, чтобы сделать 
    конкретную реализацию нужно его отнаследовать
    и переопределить методы
    _init_model - вызывается для инициализации модели T и F
    _cost - функция расчета расстояния между пространствами P и Q
    _calc_fid - функция расчета FID, для понимания когда лучше сохранить модель, как лучшую
    """
    training_env: VOXNOTModelTrainingEnvironment
    hyper_params: VOXNOTModelHyperParams
    training_hyper_params: VOXNOTModelTrainingHyperParams
    last_check_point_step: int
    last_best_step: int
    data_loader_X: LoaderSampler
    data_loader_Y: LoaderSampler
    last_loss_F_value: float
    training_name: str

    def __init__(self, device, hyper_params:VOXNOTModelHyperParams, prod_mode:bool):
        self.device = device
        self.hyper_params = hyper_params

        self.model_T = None
        self.model_F = None
        self.optim_T = None
        self.optim_F = None

        self.last_check_point_step = 0
        self.last_best_step = 0

        self._init_model(prod_mode)

        if self.model_T != None:
            print(f'Parameters: {np.sum([np.prod(p.shape) for p in self.model_T.parameters()])}')#, Parameters F: {np.sum([np.prod(p.shape) for p in self.model_F.parameters()])}')

    def _freeze(self, model):
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

    def _unfreeze(self, model):
        for p in model.parameters():
            p.requires_grad_(True)
        model.train(True)

    def load_weights(self, weights_path:str | os.PathLike):
        print(f'Load weights {weights_path}')
        self.model_T.load_state_dict(torch.load(weights_path))
        self._freeze(self.model_T)

    def load_check_point(self, check_point_step:int, model_T_state:str | os.PathLike, model_F_state:str | os.PathLike, optim_T:str | os.PathLike, optim_F:str | os.PathLike):
        self.model_T.load_state_dict(torch.load(model_T_state))
        self.model_F.load_state_dict(torch.load(model_F_state))
        self.optim_T.load_state_dict(torch.load(optim_T))
        self.optim_F.load_state_dict(torch.load(optim_F))
        self.last_check_point_step = check_point_step

    # Инициализация модели
    def _init_model(self, prod_mode):
        raise 'Not implemented'

    # Функция расстояния OT
    def _cost(self, X, Y):
        raise 'Not implemented'

    def set_cost_function(self, cost_func):
        self._cost = cost_func

    def _weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(model.weight, mode = 'fan_in', nonlinearity = 'relu')

    # Шаг обучения сети Т
    def _train_step_T(self):
        self._unfreeze(self.model_T)
        self._freeze(self.model_F)

        for iteration in range(self.training_hyper_params.iters):
            self.optim_T.zero_grad()

            X = self.data_loader_X.sample(self.training_hyper_params.batch_size)
            T_X = self.model_T(X)

            self.loss_T = self._cost(X, T_X).mean() - self.model_F(T_X).mean()
            #self.loss_T = F.mse_loss(X, T_X).mean() - self.model_F(T_X).mean()

            self.loss_T.backward()
            self.optim_T.step()

    # Шаг обучения сети F
    def _train_step_F(self):
        self._unfreeze(self.model_F)
        self._freeze(self.model_T)

        X = self.data_loader_X.sample(self.training_hyper_params.batch_size)
        Y = self.data_loader_Y.sample(self.training_hyper_params.batch_size)

        self.optim_F.zero_grad()

        self.loss_F = self.model_F(self.model_T(X)).mean() - (self.training_hyper_params.W * self.model_F(Y)).mean()

        self.last_loss_F_value = float(self.loss_F)

        self.loss_F.backward()
        self.optim_F.step()

    def predict(self, X):
        self._freeze(self.model_T)
        with torch.no_grad():
            result = torch.empty(X.shape[0], 1024) 
            
            for i in range(X.shape[0]):
                result[i] = self.model_T(X)(X[i])

        #Y = self.model_T(X).to(self.device)
        return result

    def _init_optim(self):
        self.optim_T = torch.optim.Adam(self.model_T.parameters(), lr = self.training_hyper_params.LR, weight_decay = self.training_hyper_params.weight_decay)
        self.optim_F = torch.optim.Adam(self.model_F.parameters(), lr = self.training_hyper_params.LR, weight_decay = self.training_hyper_params.weight_decay)

    def _init_data_loaders(self, dataset_X, dataset_Y):
        train_data_X, test_data_X = random_split(dataset_X, [1 - self.training_hyper_params.test_proportion, self.training_hyper_params.test_proportion])
        train_data_Y, test_data_Y = random_split(dataset_Y, [1 - self.training_hyper_params.test_proportion, self.training_hyper_params.test_proportion])

        self.data_loader_X = LoaderSampler(DataLoader(train_data_X, shuffle = False, batch_size = self.training_hyper_params.batch_size), self.device)
        self.data_loader_Y = LoaderSampler(DataLoader(train_data_Y, shuffle = False, batch_size = self.training_hyper_params.batch_size), self.device)

    def _free_mem(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_last_best_model_path(self):
        file_name = f'{self.training_name}_{self.last_best_step}.pt'
        file_name = os.path.join(self.training_env.best_point_path, f'NN_T_CP_{file_name}')
        return file_name

    def set_train_params(self,
                         training_hyper_params:VOXNOTModelTrainingHyperParams, training_env:VOXNOTModelTrainingEnvironment,
                         training_dataset_X:Dataset, training_dataset_Y:Dataset,
                         training_name:str
                         ):
        self.training_hyper_params = training_hyper_params
        self.training_env = training_env
        self.training_name = training_name

        self._init_optim()

        self.model_T.apply(self._weights_init)
        self.model_F.apply(self._weights_init)

        self._init_data_loaders(training_dataset_X, training_dataset_Y)

    def _calc_fid(self):
        raise 'Not implemented'

    def _save_check_point(self, step, last_saved_step, overwrite, out_dir):
        self._freeze(self.model_T)
        self._freeze(self.model_F)

        if overwrite == True:
            file_name_base = f'{self.training_name}_{last_saved_step}.pt'

            if os.path.isfile(os.path.join(out_dir, f'NN_T_CP_{file_name_base}')) == True:
                os.remove(os.path.join(out_dir, f'NN_T_CP_{file_name_base}'))
                os.remove(os.path.join(out_dir, f'NN_F_CP_{file_name_base}'))
                os.remove(os.path.join(out_dir, f'OPT_T_CP_{file_name_base}'))
                os.remove(os.path.join(out_dir, f'OPT_F_CP_{file_name_base}'))

        file_name_base = f'{self.training_name}_{step}.pt'

        torch.save(self.model_T.state_dict(), os.path.join(out_dir, f'NN_T_CP_{file_name_base}'))
        torch.save(self.model_F.state_dict(), os.path.join(out_dir, f'NN_F_CP_{file_name_base}'))
        torch.save(self.optim_T.state_dict(), os.path.join(out_dir, f'OPT_T_CP_{file_name_base}'))
        torch.save(self.optim_F.state_dict(), os.path.join(out_dir, f'OPT_F_CP_{file_name_base}'))

    # Тренировка модели по XNOT
    def train(self):
        best_fid = np.inf
        start_step = self.last_check_point_step

        self._free_mem()

        print(f'Starting training...')

        # Цикл тренировки
        for step in range(start_step, self.training_hyper_params.max_steps):
            # T optimization
            self._train_step_T()

            # f optimization
            self._train_step_F()

            if step % self.training_env.check_point_interval == self.training_env.check_point_interval - 1:
                self._save_check_point(step, self.last_check_point_step, self.training_env.overwrite_cp_files, self.training_env.check_point_path)
                self.last_check_point_step = step
                self._free_mem()

            if self.training_env.check_fid_interval == 0 or step % self.training_env.check_fid_interval == self.training_env.check_fid_interval - 1:
                fid = self._calc_fid()

                if (fid < best_fid and step > 500):
                    self._save_check_point(step, self.last_best_step, True, self.training_env.best_point_path)
                    self.last_best_step = step

                    print(f"Best STEP {self.training_name}: {step}, FID: {fid}")

                    best_fid = fid

            if self.training_env.write_loss_interval == 0 or step % self.training_env.write_loss_interval == self.training_env.write_loss_interval - 1:
                print(f"Step_{self.training_name}: {step}, Loss F: {self.last_loss_F_value}")
