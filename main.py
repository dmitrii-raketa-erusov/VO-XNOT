
import torch
from sources.params import VOXNOTModelHyperParams, VOXNOTModelTrainingEnvironment, VOXNOTModelTrainingHyperParams
from sources.VOXNOT import VOXNOT

# ������ ������������� � ������ ����������� ����
def main_vc():
    VOXNOT.clear_mem()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # �����-��������� ������
    model_hp = VOXNOTModelHyperParams(layers = 4, layer_size = 2048)
    # ����������� �������� ������ � ����� �� ��������� �������
    vx_prod = VOXNOT(device, 'VOXNOTMLPModel', model_hp, True)
    
    # ��������� ���� 
    vx_prod.make_conversation('����� ��� ���� � ����� � ����� ��� �����������', '����� ��� ���� � ������/�������������', '����� ���� �������� ���������')

# ������ ������������� ���������� ������
def main_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    VOXNOT.clear_mem()

    # ��������� ��������� ��������� ��� ����������(����, ��� ����������)
    environment = VOXNOTModelTrainingEnvironment('���� ��� �������� ������ �������', 0, '���� ��� ���������� chekpoints', 5000, True, 500)
    
    # �����-��������� ������
    model_hp = VOXNOTModelHyperParams(layers = 4, layer_size = 2048)

    vx = VOXNOT(device, 'VOXNOTMLPModel', model_hp, False)
    
    # ���������� � ������� W
    vx.train(True, '���� � ����� � ����� �������� ��������', '����� � ����� �������� �������', '��������� ����� ��� ������', 
             '����� ���� �������� ������ ������', VOXNOTModelTrainingHyperParams(W = 1), environment, "������_W1")
    
    # VOXNOTModelTrainingHyperParams(W = 2) - ��������� ����� �������� W ��� ���������� = 2
    vx.train(False, '', ..., VOXNOTModelTrainingHyperParams(W = 2), environment, "������_W2")
    vx.train(False, '', ..., VOXNOTModelTrainingHyperParams(W = 4), environment, "������_W4")
    vx.train(False, '', ..., VOXNOTModelTrainingHyperParams(W = 8), environment, "������_W8")

