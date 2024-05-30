import sys
import os
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
import torch

from fedlab.utils.functional import get_best_gpu

from fedavg import FedAvgClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from client_utils import err_tolerate_evaluate

class PerFedAvgClient(FedAvgClient):
    def __init__(self, cid, **kwargs) -> None:
        super().__init__(cid, **kwargs)

        self.iter_trainloader = iter(self.trainloader)

        self.hessian_free = kwargs['hessian_free']
        self.alpha = kwargs['per_alpha']
        self.beta = kwargs['per_beta']
        self.pers_epoch = kwargs['pers_epoch']
    
    def get_data_batch(self, device):
        try:
            data, target = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            data, target = next(self.iter_trainloader)
        data, target = data.to(device), target.to(device)
        return data, target
    
    def compute_grad(
        self,
        compute_model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    ):
        x, y = data_batch
        if second_order_grads:
            frz_model_params = deepcopy(compute_model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(compute_model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            compute_model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = compute_model(x)
            loss_1 = criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, compute_model.parameters())

            compute_model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = compute_model(x)
            loss_2 = criterion(logit_2, y)
            grads_2 = torch.autograd.grad(loss_2, compute_model.parameters())

            compute_model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads

        else:
            logit = compute_model(x)
            loss = criterion(logit, y)
            grads = torch.autograd.grad(loss, compute_model.parameters())
            return grads

    def train(self, global_model: torch.nn.Module):
        device = torch.device(get_best_gpu() if torch.cuda.is_available() else 'cpu')
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.train()
        self.local_model.to(device)
        if self.hessian_free:
            for _ in range(self.local_epoch):
                for _ in range(len(self.trainloader) // (2 + self.hessian_free)):
                    temp_model = deepcopy(self.local_model)
                    data_batch_1 = self.get_data_batch(device)
                    grads = self.compute_grad(temp_model, data_batch_1)
                    for param, grad in zip(temp_model.parameters(), grads):
                        param.data.sub_(self.alpha * grad)
                    
                    data_batch_2 = self.get_data_batch(device)
                    grad_1st = self.compute_grad(temp_model, data_batch_2)

                    data_batch_3 = self.get_data_batch(device)
                    grad_2nd = self.compute_grad(self.local_model, data_batch_3, grad_1st, second_order_grads=True)

                    for param, g1, g2 in zip(self.local_model.parameters(), grad_1st, grad_2nd):
                        param.data.sub_(self.beta * g1 - self.beta * self.alpha * g2)
        else:
            for _ in range(self.local_epoch):
                for _ in range(len(self.trainloader) // (2 + self.hessian_free)):
                    temp_model = deepcopy(self.local_model)
                    data_batch_1 = self.get_data_batch(device)
                    grads = self.compute_grad(temp_model, data_batch_1)
                    for param, grad in zip(temp_model.parameters(), grads):
                        param.data.sub_(self.alpha * grad)
                    
                    data_batch_2 = self.get_data_batch(device)
                    grad_1st = self.compute_grad(temp_model, data_batch_2)

                    for param, g1 in zip(self.local_model.parameters(), grad_1st):
                        param.data.sub_(self.beta * g1)
    
    # def train(self, global_model: torch.nn.Module):
    #     device = torch.device(get_best_gpu() if torch.cuda.is_available() else 'cpu')
    #     self.local_model.load_state_dict(global_model.state_dict())
    #     meta_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.beta)
    #     self.local_model.train()
    #     self.local_model.to(device)
    #     model_plus = deepcopy(self.local_model)
    #     model_minus = deepcopy(self.local_model)
    #     delta = 1e-3
    #     for _ in range(self.local_epoch):
    #         for _ in range(len(self.trainloader) // (2 + self.hessian_free)):
    #             x0, y0 = self.get_data_batch(device)
    #             frz_model = deepcopy(self.local_model)
    #             logit = self.local_model(x0)
    #             loss = self.criterion(logit, y0)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             x1, y1 = self.get_data_batch(device)
    #             logit = self.local_model(x1)
    #             loss = self.criterion(logit, y1)
    #             meta_optimizer.zero_grad()
    #             loss.backward()

    #             if self.hessian_free:
    #                 model_plus.load_state_dict(frz_model.state_dict())
    #                 model_minus.load_state_dict(frz_model.state_dict())

    #                 x2, y2 = self.get_data_batch(device)

    #                 for param, param_plus, param_minus in zip(self.local_model.parameters(), model_plus.parameters(), model_minus.parameters()):
    #                     param_plus.data += delta * param.grad
    #                     param_minus.data -= delta * param.grad
                    
    #                 logit_plus = model_plus(x2)
    #                 logit_minus = model_minus(x2)

    #                 loss_plus = self.criterion(logit_plus, y2)
    #                 loss_minus = self.criterion(logit_minus, y2)

    #                 loss_plus.backward()
    #                 loss_minus.backward()

    #                 for param, param_plus, param_minus in zip(self.local_model.parameters(), model_plus.parameters(), model_minus.parameters()):
    #                     param.grad = param.grad - self.alpha / (2 * delta) * (param_plus.grad - param_minus.grad)
    #                     param_plus.grad.zero_()
    #                     param_minus.grad.zero_()
                
    #             self.local_model.load_state_dict(frz_model.state_dict())
    #             meta_optimizer.step()

    def train_after_cluster(self, global_model: torch.nn.Module):
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

    def evaluate(self, global_model: torch.nn.Module):
        self.local_model.load_state_dict(global_model.state_dict())
        for _ in range(self.pers_epoch):
            self.train_once()
        return err_tolerate_evaluate(self.local_model, self.criterion, self.testloader)

