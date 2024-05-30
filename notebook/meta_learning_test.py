import sys
import os
import random
import logging
from copy import deepcopy
from typing import List, Tuple, Union
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import evaluate

# 将项目根目录加入环境变量
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
print(PROJECT_DIR)

from utils import read_options


def dataloader_init():
    client_trainloaders: List[DataLoader] = [cluster_partitioner.get_dataloader(cid, config['local_bs']) for cid in range(num_client)]
    client_iter_trainloaders: List[Tuple[DataLoader]] = [iter(client_trainloaders[i]) for i in range(num_client)]

    client_testloaders: List[DataLoader] = [cluster_partitioner.get_dataloader(cid, config['local_bs'], type="test") for cid in range(num_client)]

    return client_trainloaders, client_iter_trainloaders, client_testloaders

def model_init():
    client_models: List[torch.nn.Module] = [model() for _ in range(num_client)]
    client_optimizers: List[torch.optim.SGD] = [torch.optim.SGD(client_models[i].parameters(), lr=config['lr']) for i in range(num_client)]
    client_criterion: List[torch.nn.CrossEntropyLoss] = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]

    global_model: torch.nn.Module = model()
    global_optimizer: torch.optim.SGD = torch.optim.SGD(global_model.parameters(), lr=config['lr'])
    global_criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    
    return client_models, client_optimizers, client_criterion, \
        global_model, global_optimizer, global_criterion

def get_data_batch(cid, client_trainloaders, client_iter_trainloaders):
    try:
        x, y = next(client_iter_trainloaders[cid])
    except StopIteration:
        client_iter_trainloaders[cid] = iter(client_trainloaders[cid])
        x, y = next(client_iter_trainloaders[cid])

    return x.to(device), y.to(device)

def compute_grad(
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

def pers_eval(cid, client_model, client_trainloaders, client_iter_trainloaders, eval_loader):
    criterion = torch.nn.CrossEntropyLoss()
    loss_before, acc_before = evaluate(client_model, criterion, eval_loader)
    optimizer = torch.optim.SGD(client_model.parameters(), lr=config['per_alpha'])
    for _ in range(config['pers_epoch']):
        for _ in range(len(client_trainloaders[cid]) // 3):
            x, y = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
            logit = client_model(x)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    loss_after, acc_after = evaluate(client_model, criterion, eval_loader)
    return loss_before, loss_after, acc_before, acc_after

def perfedavg_1_train(cid, client_model: torch.nn.Module, client_trainloaders, client_iter_trainloaders, hessian_free=True):
    if hessian_free:
        for _ in range(config['local_ep']):
            for _ in range(len(client_trainloaders[cid]) // 3):
                temp_model = deepcopy(client_model)
                data_batch_1 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(config['per_alpha'] * grad)

                data_batch_2 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads_1st = compute_grad(temp_model, data_batch_2)

                data_batch_3 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)

                grads_2nd = compute_grad(
                    client_model, data_batch_3, v=grads_1st, second_order_grads=True
                )
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(
                    client_model.parameters(), grads_1st, grads_2nd
                ):
                    param.data.sub_(config['per_beta'] * grad1 - config['per_beta'] * config['per_alpha'] * grad2)
    else:
        for _ in range(config['local_ep']):
            for _ in range(len(client_trainloaders[cid]) // 2):
                temp_model = deepcopy(client_model)
                data_batch_1 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_1)

                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(config['per_alpha'] * grad)

                data_batch_2 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_2)

                for param, grad in zip(client_model.parameters(), grads):
                    param.data.sub_(config['per_beta'] * grad)

def perfedavg_2_train(cid, client_model: torch.nn.Module, criterion, optimizer, client_trainloaders, client_iter_trainloaders, hessian_free=True):
    client_model.train()
    meta_optimizer = torch.optim.SGD(client_model.parameters(), lr=config['per_beta'])
    model_plus = deepcopy(client_model)
    model_minus = deepcopy(client_model)
    delta = 1e-3
    for _ in range(config['local_ep']):
        for _ in range(len(client_trainloaders[cid]) // (2 + hessian_free)):
            x0, y0 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
            frz_model = deepcopy(client_model)
            logit = client_model(x0)
            loss = criterion(logit, y0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x1, y1 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
            logit = client_model(x1)
            loss = criterion(logit, y1)
            meta_optimizer.zero_grad()
            loss.backward()

            if hessian_free:
                model_plus.load_state_dict(frz_model.state_dict())
                model_minus.load_state_dict(frz_model.state_dict())

                x2, y2 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)

                for param_p, param_m, param_cur in zip(model_plus.parameters(), model_minus.parameters(), client_model.parameters()):
                    param_p.data += delta * param_cur.grad
                    param_m.data -= delta * param_cur.grad
                
                logit_plus = model_plus(x2)
                logit_minus = model_minus(x2)

                loss_plus = criterion(logit_plus, y2)
                loss_minus = criterion(logit_minus, y2)

                loss_plus.backward()
                loss_minus.backward()

                for param_p, param_m, param_cur in zip(model_plus.parameters(), model_minus.parameters(), client_model.parameters()):
                    param_cur.grad = param_cur.grad - config['lr'] / (2 * delta) * (param_p.grad - param_m.grad)
                    param_p.grad.zero_()
                    param_m.grad.zero_()
            
            client_model.load_state_dict(frz_model.state_dict())
            meta_optimizer.step()

def test(type="1"):
    client_models, client_optimizers, client_criterions, global_model, _, _ = model_init()

    for current_round in range(num_round):
        cur_train_clients = train_client_sample_stream[current_round]
        cur_test_clients = test_client_sample_stream[current_round]
        if current_round % 10 != 0:
            model_params_cache = []
            for cid in cur_train_clients:
                local_model = client_models[cid]
                local_model.load_state_dict(global_model.state_dict())
                local_optimizer = client_optimizers[cid]
                local_criterion = client_criterions[cid]
                local_model.to(device)
                if type == "1":
                    perfedavg_1_train(cid, local_model, client_trainloaders, client_iter_trainloaders)
                elif type == "2":
                    perfedavg_2_train(cid, local_model, local_criterion, local_optimizer, client_trainloaders, client_iter_trainloaders)
                model_params_cache.append(SerializationTool.serialize_model(local_model))
            aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)
            SerializationTool.deserialize_model(global_model, aggregated_model_params)
        else:
            all_metrics = {'loss_before': [], 'loss_after': [], 'acc_before': [], 'acc_after': []}
            for cid in cur_test_clients:
                local_model = client_models[cid]
                local_model.load_state_dict(global_model.state_dict())
                local_model.to(device)
                eval_loader = client_testloaders[cid]
                loss_before, loss_after, acc_before, acc_after = pers_eval(cid, local_model, client_trainloaders, client_iter_trainloaders, eval_loader)
                all_metrics['loss_before'].append(loss_before)
                all_metrics['loss_after'].append(loss_after)
                all_metrics['acc_before'].append(acc_before)
                all_metrics['acc_after'].append(acc_after)

            all_loss_before = np.mean(all_metrics['loss_before'])
            all_loss_after = np.mean(all_metrics['loss_after'])
            all_acc_before = np.mean(all_metrics['acc_before'])
            all_acc_after = np.mean(all_metrics['acc_after'])

            logger.info("round:{}, client {}, loss: {:.4f} -> {:.4f},  acc: {:.2f}% -> {:.2f}%".format(
                current_round, cur_test_clients, all_loss_before, all_loss_after, all_acc_before * 100.0, all_acc_after * 100.0
            ))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    config, cluster_partitioner, model = read_options()
    num_client = config['num_client']
    num_classes = config['num_classes']
    num_round = config['num_round']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # client_sample_stream = [
    #     random.sample(
    #         range(num_client), max(1, int(num_client * 0.2))
    #     )
    #     for _ in range(num_round)
    # ]

    train_client_sample_stream = [
        random.sample(
            range(num_client)[: int(num_client*0.8)], max(1, int(num_client * 0.2))
        )
        for _ in range(num_round)
    ]
    test_client_sample_stream = [
        random.sample(
            range(num_client)[int(num_client*0.8):], max(1, int(num_client * 0.2 * 0.8))
        )
        for _ in range(num_round)
    ]

    client_trainloaders, client_iter_trainloaders, client_testloaders = dataloader_init()

    test(type="2")