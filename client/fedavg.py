import sys
import os
from copy import deepcopy
from logging import Logger
import torch
from torch.utils.data import DataLoader

from fedlab.utils.functional import get_best_gpu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from client_utils import err_tolerate_evaluate

class FedAvgClient():
    def __init__(self, cid, **kwargs) -> None:
        self.cid: int = cid
        self.logger: Logger = kwargs['logger']

        self.per_model: torch.nn.Module = None

        self.local_model: torch.nn.Module  = deepcopy(kwargs['model_template'])
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(self.local_model.parameters(), lr=kwargs['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()

        self.trainloader: DataLoader = kwargs['trainloader']
        self.testloader: DataLoader = kwargs['testloader']

        self.local_epoch: int = kwargs['local_epoch']
    
    def train(self, global_model: torch.nn.Module):
        device = torch.device(get_best_gpu() if torch.cuda.is_available() else 'cpu')
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.train()
        self.local_model.to(device)
        for _ in range(self.local_epoch):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def train_once(self):
        device = torch.device(get_best_gpu() if torch.cuda.is_available() else 'cpu')
        self.local_model.train()
        self.local_model.to(device)
        for data, target in self.trainloader:
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.local_model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, eval_model: torch.nn.Module):
        self.local_model.load_state_dict(eval_model.state_dict())
        return err_tolerate_evaluate(self.local_model, self.criterion, self.testloader)

    def evaluate_without_train(self, eval_model: torch.nn.Module):
        self.local_model.load_state_dict(eval_model.state_dict())
        return err_tolerate_evaluate(self.local_model, self.criterion, self.testloader)