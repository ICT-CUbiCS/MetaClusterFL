import sys
import os
import random
import time
import math
import logging
import threading
import concurrent.futures
from copy import deepcopy
from typing import List, Tuple, Union, Dict
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.functional import evaluate, get_best_gpu

# 将项目根目录加入环境变量
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
print(PROJECT_DIR)

from utils import read_options
from client_utils import detail_evaluate, err_tolerate_evaluate

def dataloader_init():
    client_trainloaders: List[DataLoader] = [cluster_partitioner.get_dataloader(cid, config['local_bs']) for cid in range(num_client)]
    client_iter_trainloaders: List[Tuple[DataLoader]] = [iter(client_trainloaders[i]) for i in range(num_client)]

    client_testloaders: List[DataLoader] = [cluster_partitioner.get_dataloader(cid, config['local_bs'], type="test") for cid in range(num_client)]

    return client_trainloaders, client_iter_trainloaders, client_testloaders

def global_model_init():
    global_model: torch.nn.Module = model()
    global_optimizer: torch.optim.SGD = torch.optim.SGD(global_model.parameters(), lr=config['lr'])
    global_criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    return global_model, global_optimizer, global_criterion

def model_init():
    client_models: List[torch.nn.Module] = [model() for _ in range(num_client)]
    client_optimizers: List[torch.optim.SGD] = [torch.optim.SGD(client_models[i].parameters(), lr=config['lr']) for i in range(num_client)]
    client_criteria: List[torch.nn.CrossEntropyLoss] = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]

    global_model: torch.nn.Module = model()
    global_optimizer: torch.optim.SGD = torch.optim.SGD(global_model.parameters(), lr=config['lr'])
    global_criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    
    return client_models, client_optimizers, client_criteria, global_model, global_optimizer, global_criterion

def train(cid: int, model: torch.nn.Module, optimizer: torch.optim.SGD, criterion: torch.nn.CrossEntropyLoss):
    model.to(device)
    model.train()
    train_loader = cluster_partitioner.get_dataloader(cid, config['local_bs'])
    for _ in range(config['local_ep']):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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

def perfedavg_1_train_scaffold(cid, client_model: torch.nn.Module, client_trainloaders, client_iter_trainloaders, c_global: List[torch.Tensor], c_local: List[torch.Tensor], hessian_free=True):
    client_model.to(device)
    # first_grads = None
    if hessian_free:
        for _ in range(config['local_ep']):
            for _ in range(len(client_trainloaders[cid]) // 3):
                temp_model = deepcopy(client_model)
                data_batch_1 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(config['per_alpha'] * grad)
                
                # if first_grads is None:
                #     first_grads = torch.cat([grad.view(-1) for grad in grads])

                data_batch_2 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads_1st = compute_grad(temp_model, data_batch_2)

                data_batch_3 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)

                grads_2nd = compute_grad(
                    client_model, data_batch_3, v=grads_1st, second_order_grads=True
                )
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2, c_l, c_g in zip(
                    client_model.parameters(), grads_1st, grads_2nd, c_local, c_global
                ):
                    param.data.sub_(config['per_beta'] * grad1 - config['per_beta'] * config['per_alpha'] * grad2 - config['per_beta'] * c_l + config['per_beta'] * c_g)
    else:
        for _ in range(config['local_ep']):
            for _ in range(len(client_trainloaders[cid]) // 2):
                temp_model = deepcopy(client_model)
                data_batch_1 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(config['per_alpha'] * grad)
                
                # if first_grads is None:
                #     first_grads = torch.cat([grad.view(-1) for grad in grads])

                data_batch_2 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_2)

                for param, grad in zip(client_model.parameters(), grads):
                    param.data.sub_(config['per_beta'] * grad)


def perfedavg_1_train(cid, client_model, client_trainloaders, client_iter_trainloaders, hessian_free=True):
    client_model.to(device)
    # first_grads = None
    if hessian_free:
        for _ in range(config['local_ep']):
            for _ in range(len(client_trainloaders[cid]) // 3):
                temp_model = deepcopy(client_model)
                data_batch_1 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(config['per_alpha'] * grad)
                
                # if first_grads is None:
                #     first_grads = torch.cat([grad.view(-1) for grad in grads])

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
                
                # if first_grads is None:
                #     first_grads = torch.cat([grad.view(-1) for grad in grads])

                data_batch_2 = get_data_batch(cid, client_trainloaders, client_iter_trainloaders)
                grads = compute_grad(temp_model, data_batch_2)

                for param, grad in zip(client_model.parameters(), grads):
                    param.data.sub_(config['per_beta'] * grad)


def evaluate_clusters(client_cluster, model_cluster):
    # print("evaluate_clusters:")
    num_classes = config['num_classes']
    global_eval = np.zeros(num_classes + 1)
    for cluster_cids, cluster_model in zip(client_cluster, model_cluster):
        # print("cluster_cids: {}".format(cluster_cids))
        logger.info("cluster_cids: {}".format(cluster_cids))
        test_loader = cluster_partitioner.get_cluster_dataloader(cluster_cids, config['local_bs'], "test")
        result = np.array(detail_evaluate(cluster_model, torch.nn.CrossEntropyLoss(), test_loader, num_classes))
        global_eval += result
    global_eval /= len(client_cluster)
    global_eval = np.round(global_eval, 4)
    # print("global_eval: {}".format(global_eval.tolist()))
    logger.info("global_eval: {}".format(global_eval.tolist()))
    return global_eval

def create_cluster_model(cluster: List[int], client_models: List[torch.nn.Module]):
    if len(cluster) == 1:
        return client_models[cluster[0]]
    model_param_lsit = [SerializationTool.serialize_model(client_models[cid]) for cid in cluster]
    avg_param = Aggregators.fedavg_aggregate(model_param_lsit)
    cluster_model = model()
    SerializationTool.deserialize_model(cluster_model, avg_param)
    return cluster_model

def remove_clients_from_cluster(client_cluster, model_cluster, similarity_matrix, client_models):
    client_cluster_new = []
    model_cluster_new = []
    for index, cluster in enumerate(client_cluster):
        if len(cluster) < 2:
            client_cluster_new.append(cluster)
            model_cluster_new.append(model_cluster[index])
            continue
        cluster_similarity_matrix = similarity_matrix[cluster, :][:, cluster]
        mask = np.where((cluster_similarity_matrix > -1.0) & (cluster_similarity_matrix < 0.0))
        if not np.any(mask):
            client_cluster_new.append(cluster)
            model_cluster_new.append(model_cluster[index])
            continue
        outlier_indices = np.unique(mask)
        cluster = np.array(cluster)
        outlier_clients = cluster[outlier_indices]
        reminder_clients = np.delete(cluster, outlier_indices)
        if len(reminder_clients) > 0:
            client_cluster_new.append(reminder_clients.tolist())
            model_cluster_new.append(create_cluster_model(reminder_clients, client_models))
        for cid in outlier_clients:
            client_cluster_new.append([cid])
            model_cluster_new.append(client_models[cid])
        # print("remove {} from {}".format(outlier_clients, cluster))
        logger.info("remove {} from {}".format(outlier_clients.tolist(), cluster.tolist()))
        
        # mask = np.where(cluster_similarity_matrix > -1.0)
        # if not np.any(mask):
        #     client_cluster_new.append(cluster)
        #     model_cluster_new.append(model_cluster[index])
        #     continue
        # min_similarity = np.min(cluster_similarity_matrix[mask])
        # if min_similarity < 0:
        #     clustering = AgglomerativeClustering(
        #         affinity="precomputed", linkage="complete"
        #     ).fit(-cluster_similarity_matrix)
        #     cluster_1 = [cluster[i] for i in np.argwhere(clustering.labels_ == 0).flatten()]
        #     cluster_2 = [cluster[i] for i in np.argwhere(clustering.labels_ == 1).flatten()]
        #     if len(cluster_1) == 0 or len(cluster_2) == 0:
        #         client_cluster_new.append(cluster)
        #         model_cluster_new.append(model_cluster[index])
        #     else:
        #         print("split {} into {} and {}".format(cluster, cluster_1, cluster_2))
        #         model_1 = create_cluster_model(cluster_1, client_models)
        #         model_2 = create_cluster_model(cluster_2, client_models)
        #         client_cluster_new += [cluster_1, cluster_2]
        #         model_cluster_new += [model_1, model_2]
        # else:
        #     client_cluster_new.append(cluster)
        #     model_cluster_new.append(model_cluster[index])
    return client_cluster_new, model_cluster_new

def improve_test():
    client_models, client_optimizers, client_criteria, global_model, _, _ = model_init()
    similarity_matrix = np.full((num_client, num_client), -1.0, dtype=np.float32)
    memory_matrix = np.full((num_client, num_client), 0, dtype=np.int32)
    client_cluster = [[i] for i in range(num_client)]
    model_cluster = [model() for i in range(num_client)]
    client_cluster_model = {i: model_cluster[i] for i in range(num_client)}

    no_merge_count = 0

    for current_round in range(num_round):
        grads_list: List[torch.Tensor] = [None] * num_client
        selected_clients = client_sample_stream[current_round]
        # print("current round: {}, selected clients: {}".format(current_round, selected_clients))
        logger.info("current round: {}, selected clients: {}".format(current_round, selected_clients))
        # if no_merge_count < 10:
        if True:
            for cid in selected_clients:
                local_model: torch.nn.Module = client_models[cid]
                local_model.load_state_dict(global_model.state_dict())
                local_optimizer = client_optimizers[cid]
                local_criterion = client_criteria[cid]
                
                # param_before = SerializationTool.serialize_model(local_model).detach()
                # # train(cid, local_model, local_optimizer, local_criterion)
                # perfedavg_1_train(cid, local_model, client_trainloaders, client_iter_trainloaders)
                # param_after = SerializationTool.serialize_model(local_model).detach()
                # grads = param_before - param_after

                grads = perfedavg_1_train(cid, local_model, client_trainloaders, client_iter_trainloaders, False)

                grads_list[cid] = grads
            
            for i, cid_i in enumerate(selected_clients):
                for j, cid_j in enumerate(selected_clients[i+1:], i+1):
                    if grads_list[cid_i] is not None and grads_list[cid_j] is not None:
                        similarity_score = torch.cosine_similarity(
                            grads_list[cid_i], grads_list[cid_j], dim=0, eps=1e-12
                        ).item()
                        similarity_matrix[cid_i, cid_j] = similarity_score
                        similarity_matrix[cid_j, cid_i] = similarity_score

                        memory_matrix[cid_i, cid_j] = current_round
                        memory_matrix[cid_j, cid_i] = current_round
                        
            # 从 cluster 中删除 client (二分)
            client_cluster, model_cluster = remove_clients_from_cluster(
                client_cluster, model_cluster, similarity_matrix, client_models
            )
            
            # 合并 cluster
            max_min_similarity = float("-inf")
            cluster1_index, cluster2_index = -1, -1
            cross_max_similarity = None
            within_min_similarity = None
            cross_min_similarity = None
            for i, cluster_i in enumerate(client_cluster):
                for j, cluster_j in enumerate(client_cluster[i+1:], i+1):
                    
                    if len(cluster_i) == 1 and len(cluster_j) == 1:
                        min_similarity = similarity_matrix[cluster_i[0], cluster_j[0]]
                    else:
                        mask = similarity_matrix[cluster_i][:, cluster_j] > -1.0
                        if not np.any(mask):
                            continue
                        min_similarity = np.min(similarity_matrix[cluster_i][:, cluster_j][mask])
                    
                    if min_similarity <= max_min_similarity:
                        continue

                    if len(cluster_i) > 1 and len(cluster_j) > 1:
                        mask1 = similarity_matrix[cluster_i][:, cluster_i] > -1.0
                        mask2 = similarity_matrix[cluster_j][:, cluster_j] > -1.0
                        min_cluster_i = np.min(similarity_matrix[cluster_i][:, cluster_i][mask1]) if np.any(mask1) else float('inf')
                        min_cluster_j = np.min(similarity_matrix[cluster_j][:, cluster_j][mask2]) if np.any(mask2) else float('inf')
                        if math.isinf(min_cluster_i) and math.isinf(min_cluster_j):
                            continue
                        within_min_similarity = min(min_cluster_i, min_cluster_j)

                    max_min_similarity = min_similarity
                    cluster1_index, cluster2_index = i, j
                    cross_max_similarity =  np.max(similarity_matrix[cluster_i][:, cluster_j])
                    cross_min_similarity =  min_similarity
                
            # print("cross_min_similarity: {}, cross_max_similarity: {}, within_min_similarity: {}".format(
            #     round(float(cross_min_similarity), 4), round(float(cross_max_similarity), 4), within_min_similarity
            # ))
            logger.info("cross_min_similarity: {}, cross_max_similarity: {}, within_min_similarity: {}".format(
                round(float(cross_min_similarity), 4), round(float(cross_max_similarity), 4), within_min_similarity
            ))
            if cross_min_similarity > 0 and (within_min_similarity is None or cross_max_similarity > within_min_similarity):
                # print("merge {} and {}".format(client_cluster[cluster1_index], client_cluster[cluster2_index]))
                logger.info("merge {} and {}".format(client_cluster[cluster1_index], client_cluster[cluster2_index]))
                cluster1 = client_cluster[cluster1_index]
                cluster2 = client_cluster[cluster2_index]
                client_cluster.remove(cluster1)
                client_cluster.remove(cluster2)
                client_cluster.append(cluster1 + cluster2)
                
                model1 = model_cluster[cluster1_index]
                model2 = model_cluster[cluster2_index]
                model_cluster.remove(model1)
                model_cluster.remove(model2)
                cluster_model = create_cluster_model(cluster1 + cluster2, client_models)
                model_cluster.append(cluster_model)
                
                for cid in cluster1 + cluster2:
                    client_cluster_model[cid] = cluster_model

                no_merge_count = 0
            
            else:
                no_merge_count += 1

            out_memory_indices = np.where((current_round - memory_matrix) > 10)
            similarity_matrix[out_memory_indices] = -1
            
            selected_clients_param_list = [SerializationTool.serialize_model(client_models[cid]) for cid in selected_clients]
            selected_clients_avg_param = Aggregators.fedavg_aggregate(selected_clients_param_list)
            SerializationTool.deserialize_model(global_model, selected_clients_avg_param)
        
        else:
            for cid in selected_clients:
                local_model: torch.nn.Module = client_models[cid]
                local_model.load_state_dict(client_cluster_model[cid].state_dict())
                local_optimizer = client_optimizers[cid]
                local_criterion = client_criteria[cid]
                
                train(cid, local_model, local_optimizer, local_criterion)
            
            selected_clients_set = set(selected_clients)
            for index, cluster in enumerate(client_cluster):
                cluster_set = set(cluster)
                intersection = selected_clients_set.intersection(cluster_set)
                if len(intersection) > 0:
                    # print("update cluster {} using clients {}".format(cluster, intersection))
                    logger.info("update cluster {} using clients {}".format(cluster, intersection))
                    param_list = [SerializationTool.serialize_model(client_models[cid]) for cid in intersection]
                    avg_param = Aggregators.fedavg_aggregate(param_list)
                    SerializationTool.deserialize_model(model_cluster[index], avg_param)

        if current_round % 20 == 0:
            evaluate_clusters(client_cluster, model_cluster)

def create_similarity_matrix(grads: List[torch.Tensor]):
    similarity_matrix = np.full((len(grads), len(grads)), 0, dtype=np.float16)
    for i, grad_i in enumerate(grads):
        for j, grad_j in enumerate(grads[i + 1:], i+1):
            similarity_score = torch.cosine_similarity(grad_i, grad_j, dim=0).item()
            similarity_matrix[i][j] = similarity_score
            similarity_matrix[j][i] = similarity_score
    return similarity_matrix

def meta_cluster_test():
    client_models, client_optimizers, client_criterions, global_model, _, _ = model_init()
    model_cluster = [model() for _ in range(num_client)]

    cluster_info = {
        cluster_id: {
            "clients": [cluster_id],
            "model": model_cluster[cluster_id],
            "grad": None,
            "similarity_record": 0.0,

            "train_clients": [],
            "train_grads": []
        }
        for cluster_id in range(num_client)
    }

    client_info = {
        cid: {
            "local_model": client_models[cid],
            "local_optimizer": client_optimizers[cid],
            "local_criterion": client_criterions[cid],
            "grad": None,

            "cluster": cluster_info[cid],
        }
        for cid in range(num_client)
    }

    unique_num = num_client

    for current_round in range(num_round):
        selected_clients = client_sample_stream[current_round]
        logger.info("round: {}, selected clients: {}".format(current_round, selected_clients))
        for cid in selected_clients:
            local_model: torch.nn.Module = client_info[cid]["local_model"]
            local_model.load_state_dict(global_model.state_dict())
            local_optimizer = client_info[cid]["local_optimizer"]
            local_criterion = client_info[cid]["local_criterion"]

            grad = perfedavg_1_train(cid, local_model, client_trainloaders, client_iter_trainloaders)
            # param_before = SerializationTool.serialize_model(local_model)
            # train(cid, local_model, local_optimizer, local_criterion)
            # param_after = SerializationTool.serialize_model(local_model)
            # grad = param_before - param_after

            client_info[cid]["grad"] = grad
            client_info[cid]["cluster"]["train_clients"].append(cid)
            client_info[cid]["cluster"]["train_grads"].append(client_info[cid]["grad"])
        
        # 聚类内 client 相似性计算
        participant_cluster = []
        dispear_cluster = []
        seperate_clients = []
        for cluster_id, info in cluster_info.items():
            if len(info["train_clients"]) == 0:
                continue
            elif len(info["train_clients"]) == 1:
                info["grad"] = info["train_grads"][0]
                participant_cluster.append(cluster_id)
            else:
                inner_similarity_matrix = create_similarity_matrix(info["train_grads"])
                outlier_index = np.where((inner_similarity_matrix > -1.0) & (inner_similarity_matrix < 0))
                # 处理 cluster 中离群的 client, 依据是 grad 相似性小于 0
                if np.any(outlier_index):
                    outlier_index = np.sort(np.unique(outlier_index))[::-1]
                    seperate_clients = [info["train_clients"][index] for index in outlier_index]
                    for cid in seperate_clients:
                        cluster_info[cluster_id]['clients'].remove(cid)
                        cluster_info[cluster_id]["train_clients"].remove(cid)
                    for index in outlier_index:
                        info["train_grads"].pop(index)
                    logger.info("round: {}, cluster: {}, seperate clients: {}".format(current_round, cluster_info[cluster_id]['clients'], seperate_clients))
                
                # 处理 info 中剩下的 client
                if len(info["clients"]) == 0:
                    dispear_cluster.append(cluster_id)
                elif len(info["clients"]) == 1:
                    info["grad"] = info["train_grads"][0]
                    participant_cluster.append(cluster_id)
                else:
                    info["grad"] = Aggregators.fedavg_aggregate(info["train_grads"])
                    participant_cluster.append(cluster_id)

        # 处理空的 cluster
        for cluster_id in dispear_cluster:
            cluster_info.pop(cluster_id)
        
        # 处理离群的 client, 为其创建新的 cluster
        for cid in seperate_clients:
            cluster_info.setdefault(unique_num, {
                "clients": [cid],
                "model": client_info[cid]['local_model'],
                "grad": grad,
                "similarity_record": 0.0,

                "train_clients": [cid],
                "train_grads": [client_info[cid]['grad']]
            })
            participant_cluster.append(unique_num)
            client_info[cid]["cluster"] = cluster_info[unique_num]
            unique_num += 1

        # 聚类间 client 相似性计算
        participant_grads = [cluster_info[cluster_id]["grad"] for cluster_id in participant_cluster]
        participant_cluster = np.array(participant_cluster)
        ## 计算聚类之间的相似性, 利用当前训练的 client 的平均 grad 计算
        participant_similarity_matrix = create_similarity_matrix(participant_grads)
        participant_max_similarity = np.max(participant_similarity_matrix)
        if participant_max_similarity > 0:  # 存在可能相似的聚类
            combine_similarity_threshold = participant_max_similarity * 0.9 #TODO 这里可以设置一个随 current_round 增加而增加的阈值
            ## 找出相似阈值内的聚类对, 并按照相似性排序
            indices = np.where(participant_similarity_matrix > combine_similarity_threshold)
            elements = participant_similarity_matrix[indices]
            coordinates = list(zip(indices[0], indices[1]))
            sorted_indices = np.argsort(elements)[::-1]
            sorted_elements = elements[sorted_indices]
            sorted_coordinates = [coordinates[index] for index in sorted_indices]
            ## 遍历相似聚类对, 尝试合并
            for element, coordinate in zip(sorted_elements, sorted_coordinates):
                combine_cluster1 = participant_cluster[coordinate[0]]
                combine_cluster2 = participant_cluster[coordinate[1]]
                if cluster_info.get(combine_cluster1) is None or cluster_info.get(combine_cluster2) is None:
                    continue
                ## 如果两个聚类之间的相似度小于聚类以往聚合的平均值, 则不合并
                avg_similarity = (cluster_info[combine_cluster1]["similarity_record"] + cluster_info[combine_cluster2]["similarity_record"]) / 2
                if element < avg_similarity * 0.8:
                    continue
                combine_cluster1_model = cluster_info[combine_cluster1]["model"]
                combine_cluster2_model = cluster_info[combine_cluster2]["model"]
                new_cluster_model = model()
                combine_cluster1_model_param = SerializationTool.serialize_model(combine_cluster1_model).detach()
                combine_cluster2_model_param = SerializationTool.serialize_model(combine_cluster2_model).detach()
                new_cluster_model_param = Aggregators.fedavg_aggregate([combine_cluster1_model_param, combine_cluster2_model_param])
                SerializationTool.deserialize_model(new_cluster_model, new_cluster_model_param)
                new_cluster_clients = cluster_info[combine_cluster1]["clients"] + cluster_info[combine_cluster2]["clients"]
                cluster_info.setdefault(unique_num, {
                    "clients": new_cluster_clients,
                    "model": new_cluster_model,
                    "grad": None,
                    "similarity_record": (element + 2 * avg_similarity) / 3 if avg_similarity > 0 else element,

                    "train_clients": [],
                    "train_grads": []
                })
                for client in new_cluster_clients:
                    client_info[client]["cluster"] = cluster_info[unique_num]
                unique_num += 1
                logger.info("combine cluster {} and cluster {}, {} and {} with the similarity {}".format(
                    cluster_info[combine_cluster1]["clients"], cluster_info[combine_cluster2]["clients"], 
                    cluster_info[combine_cluster1]['train_clients'], cluster_info[combine_cluster2]['train_clients'],
                    round(element, 4)
                ))
                cluster_info.pop(combine_cluster1)
                cluster_info.pop(combine_cluster2)
        
        # 处理剩余的 cluster
        for cluster_id in participant_cluster:
            info = cluster_info.get(cluster_id)
            if info is None:
                continue
            info["grad"] = None
            info["train_clients"] = []
            info["train_grads"] = []

def plot_hot_map(similarity_matrix: np.ndarray, current_round: int, call_name: str):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    ax.set_title("round {} similarity hot map".format(current_round + 1))
    # fig.colorbar(mappable=ax.images[0], ax=ax)
    fig.tight_layout()
    img_dir = os.path.join(PROJECT_DIR, "result", "plot", "cluster_test", call_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_path = os.path.join(img_dir, "round_{0:0>3}.png".format(current_round + 1))
    fig.savefig(img_path, dpi=600)
    plt.close()

def agglomerative_clients(similarity_matrix: np.ndarray, num_clusters: int = 10):
    # max_similarity = np.max(similarity_matrix)
    # threshold = max_similarity * 0.95
    # upper_triangular = np.triu(similarity_matrix, k=1)
    # indices = np.where(upper_triangular > threshold)
    # elements = similarity_matrix[indices]
    # coordinates = list(zip(indices[0], indices[1]))
    # sorted_indices = np.argsort(elements)[::-1]
    # sorted_elements = elements[sorted_indices]
    # sorted_coordinates = [coordinates[index] for index in sorted_indices]
    # return sorted_elements, sorted_coordinates
    
    # cluster = AgglomerativeClustering(
    #     n_clusters=num_clusters, metric="precomputed", linkage='average'
    # )
    cluster = SpectralClustering(n_clusters=num_clusters, affinity="precomputed")
    cluster.fit(similarity_matrix)
    # 统计每个聚类对应similarity_matrix中的行列
    cluster_dict = {}
    for index, label in enumerate(cluster.labels_):
        cluster_dict.setdefault(label, [])
        cluster_dict[label].append(index)
    return cluster_dict

def seek_within_min_similarity(similarity_matrix, clients_cluster_1, clients_cluster_2):
    mask1 = np.eye(len(clients_cluster_1), dtype=bool)
    mask2 = np.eye(len(clients_cluster_2), dtype=bool)
    masked_cluster1_matrix = similarity_matrix[np.ix_(clients_cluster_1, clients_cluster_1)][~mask1]
    masked_cluster2_matrix = similarity_matrix[np.ix_(clients_cluster_2, clients_cluster_2)][~mask2]
    if masked_cluster1_matrix.size > 0 and masked_cluster2_matrix.size > 0:
        return min(np.min(masked_cluster1_matrix), np.min(masked_cluster2_matrix))
    elif masked_cluster1_matrix.size > 0:
        return np.min(masked_cluster1_matrix)
    elif masked_cluster2_matrix.size > 0:
        return np.min(masked_cluster2_matrix)
    else:
        return 0

def separate_clients(similarity_matrix: np.ndarray, cluster_client: List[List[int]]):
    new_cluster_client = []
    for clients in cluster_client:
        if len(clients) == 1:
            new_cluster_client.append(clients)
            continue
        clients = np.array(clients)
        clients_similarity_matrix = similarity_matrix[clients][:, clients]
        clients_clustering = SpectralClustering(n_clusters=2, affinity="precomputed")
        clients_clustering.fit(clients_similarity_matrix)
        clients_cluster_1 = clients[clients_clustering.labels_ == 0]
        clients_cluster_2 = clients[clients_clustering.labels_ == 1]
        cross_max_similarity = np.max(similarity_matrix[np.ix_(clients_cluster_1, clients_cluster_2)])
        cross_min_similarity = np.min(similarity_matrix[np.ix_(clients_cluster_1, clients_cluster_2)])
        within_min_similarity = seek_within_min_similarity(similarity_matrix, clients_cluster_1, clients_cluster_2)
        logger.info("cross_max_similarity: {}, cross_min_similarity: {}, within_min_similarity: {} between {} and {}".format(
            round(cross_max_similarity, 4), round(cross_min_similarity, 4), round(within_min_similarity, 4),
            clients_cluster_1, clients_cluster_2
        ))
        if cross_max_similarity <= within_min_similarity or cross_min_similarity < 0:
            logger.info("separate clients {} into {} and {}".format(clients, clients_cluster_1, clients_cluster_2))
            new_cluster_client.append(clients_cluster_1)
            new_cluster_client.append(clients_cluster_2)
        else:
            new_cluster_client.append(clients)
    
    return new_cluster_client

def train_once(train_model: torch.nn.Module, trainloader: DataLoader):
    optimizer = torch.optim.SGD(train_model.parameters(), lr=config['lr'], momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train_model.to(device)
    train_model.train()
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = train_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def meta_client_process(client_id: int, global_model: torch.nn.Module, trainloaders, iter_trainloaders):

    # meta training
    train_model = deepcopy(global_model)
    perfedavg_1_train(client_id, train_model, trainloaders, iter_trainloaders)
    meta_train_param = SerializationTool.serialize_model(train_model).detach()

    tmp_model = deepcopy(global_model)
    tmp_param_before = SerializationTool.serialize_model(tmp_model).detach()
    # tmp_param_before = tmp_model.get_param_vector()
    train_once(tmp_model, trainloaders[client_id])
    tmp_param_after = SerializationTool.serialize_model(tmp_model).detach()
    # tmp_param_after = tmp_model.get_param_vector()
    tmp_grad = tmp_param_before - tmp_param_after
    tmp_grad_norm = torch.norm(tmp_grad)
    if tmp_grad_norm != 0:
        tmp_grad = tmp_grad / tmp_grad_norm

    return client_id, tmp_grad, meta_train_param

def meta_client_process_scaffold(client_id: int, global_model: torch.nn.Module, trainloaders, iter_trainloaders, c_global, c_local):

    # meta training
    train_model = deepcopy(global_model)
    c_global_gpu = [c.to(device) for c in c_global]
    c_local_gpu = [c.to(device) for c in c_local]
    # perfedavg_1_train_scaffold(client_id, train_model, trainloaders, iter_trainloaders, c_global, c_local)
    perfedavg_1_train_scaffold(client_id, train_model, trainloaders, iter_trainloaders, c_global_gpu, c_local_gpu)
    meta_train_param = SerializationTool.serialize_model(train_model).detach()

    # 
    tmp_model = deepcopy(global_model)
    tmp_param_before = SerializationTool.serialize_model(tmp_model).detach()
    # tmp_param_before = tmp_model.get_param_vector().detach()
    train_once(tmp_model, trainloaders[client_id])
    tmp_param_after = SerializationTool.serialize_model(tmp_model).detach()
    # tmp_param_after = tmp_model.get_param_vector().detach()
    tmp_grad = tmp_param_before - tmp_param_after
    tmp_grad_norm = torch.norm(tmp_grad)
    if tmp_grad_norm != 0:
        tmp_grad = tmp_grad / tmp_grad_norm
    c_plus = [deepcopy(param.grad.data).to('cpu') for param in tmp_model.parameters()]
    c_delta = [c_p - c_l for c_p, c_l in zip(c_plus, c_local)]

    return client_id, c_plus, c_delta, tmp_grad, meta_train_param

def meta_client_evaluate(clients: List[int], global_model: torch.nn.Module, trainloaders, testloaders):
    cluster_personal_model = deepcopy(global_model)
    # for client_id in clients:
    #     train_once(cluster_personal_model, trainloaders[client_id])
    
    # client_model_params = []
    clients_model_param_sum = torch.cat([torch.zeros_like(param.data.view(-1)) for param in global_model.parameters()])
    for client_id in clients:
        client_model = deepcopy(cluster_personal_model)
        train_once(client_model, trainloaders[client_id])
        clients_model_param_sum += SerializationTool.serialize_model(client_model).detach()
        # client_model_params.append(SerializationTool.serialize_model(client_model).detach())
    # client_model_avg_param = Aggregators.fedavg_aggregate(client_model_params)
    client_model_avg_param = clients_model_param_sum / len(clients)
    SerializationTool.deserialize_model(cluster_personal_model, client_model_avg_param)
    
    loss, top1, top5 = [], [], []
    for client_id in clients:
        loss_, top1_, top5_ = err_tolerate_evaluate(cluster_personal_model, torch.nn.CrossEntropyLoss(), testloaders[client_id])
        loss.append(loss_)
        top1.append(top1_)
        top5.append(top5_)
    return np.mean(loss), np.mean(top1), np.mean(top5)


def meta_cluster_test2():
    # client_models, client_optimizers, client_criteria, global_model, global_optimizer, global_criterion = model_init()
    global_model, global_optimizer, global_criterion = global_model_init()
    # c_global = [torch.zeros_like(param) for param in global_model.parameters()]
    # c_local: Dict[List[torch.Tensor]] = {}
    client_similarity_matrix = np.full((num_client, num_client), 0, dtype=np.float16)
    cluster_clients: List[List[int]] = [[i] for i in range(num_client)]
    num_cluster = 4
    plot_dir_name = "meta_cluster_test2_with_scaffold_and_full_layers_vgg"
    for current_round in range(num_round):
        selected_clients = np.array(client_sample_stream[current_round])
        logger.info("round {} select clients {}".format(current_round + 1, selected_clients))

        if (current_round + 1) % 10 == 0:
            logger.info("round {}: plot the similarity hot map".format(current_round + 1))
            thread = threading.Thread(target=plot_hot_map, args=(deepcopy(client_similarity_matrix), current_round, plot_dir_name))
            thread.start()

            cluster_start_time = time.time()
            cluster_result = agglomerative_clients(client_similarity_matrix, num_cluster)
            new_cluster_clients = []
            for cluster_label, clients in cluster_result.items():
                cluster_min_similarity = np.min(client_similarity_matrix[clients][:, clients])
                cluster_max_similarity = np.max(client_similarity_matrix[clients][:, clients])
                if current_round + 1 <= 100:
                    logger.info("round {}: cluster {} has {} clients, min similarity is {}, max similarity is {}".format(
                        current_round + 1, cluster_label, len(clients), np.round(cluster_min_similarity, 4), np.round(cluster_max_similarity, 4)
                    ))
                else:
                    logger.info("round {}: cluster {} has {} clients, min similarity is {}, max similarity is {}".format(
                        current_round + 1, cluster_label, clients, np.round(cluster_min_similarity, 4), np.round(cluster_max_similarity, 4)
                    ))

                new_cluster_clients.append(clients)
            cluster_clients = new_cluster_clients
            
            if current_round + 1 >= 100:
                new_cluster_clients = separate_clients(client_similarity_matrix, cluster_clients)
                # num_cluster += len(new_cluster_clients) - len(cluster_clients)
                if len(new_cluster_clients) > len(cluster_clients):
                    num_cluster += 1
                cluster_clients = new_cluster_clients
            cluster_end_time = time.time()
            logger.info("round {}: cluster time is {}".format(current_round + 1, cluster_end_time - cluster_start_time))

            loss_cache, top1_acc_cache, top5_acc_cache = [], [], []
            ## 多线程处理
            # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            #     submit_tasks = []
            #     for clients in cluster_clients:
            #         submit_tasks.append(executor.submit(
            #             meta_client_evaluate, clients, global_model, client_trainloaders, client_testloaders
            #         ))
                
            #     for future in concurrent.futures.as_completed(submit_tasks):
            #         loss_, acc_ = future.result()
            #         loss_cache.append(loss_)
            #         acc_chache.append(acc_)

            for clients in cluster_clients:
                # loss_, acc_ = meta_client_evaluate(clients, global_model, client_trainloaders, client_testloaders)
                loss_, top1_acc_, top5_acc_ = meta_client_evaluate(clients, global_model, client_trainloaders, client_testloaders)
                loss_cache.append(loss_)
                top1_acc_cache.append(top1_acc_)
                top5_acc_cache.append(top5_acc_)
            logger.info("round {}: cluster loss is {}, cluster top1 acc is {}, cluster top5 acc is {}".format(
                current_round + 1, np.round(np.mean(loss_cache), 4), np.round(np.mean(top1_acc_cache), 4), np.round(np.mean(top5_acc_cache), 4)
            ))

        else:
            selected_clients_grads = []
            c_delta_cache = []
            local_model_param_cache = []
            
            ## 多线程处理
            # train_clients = []
            # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            #     submit_task = []
            #     for client_id in selected_clients:
            #         if client_id not in c_local.keys():
            #             c_local[client_id] = [torch.zeros_like(c, device=device) for c in c_global]
            #         submit_task.append(executor.submit(
            #             meta_client_process, 
            #             client_id, global_model, client_trainloaders, client_iter_trainloaders, c_global, c_local[client_id]
            #         ))
                
            #     for future in concurrent.futures.as_completed(submit_task):
            #         client_id, c_plus, c_delta, tmp_grad, meta_train_param = future.result()
            #         train_clients.append(client_id)
            #         c_local[client_id] = c_plus
            #         c_delta_cache.append(c_delta)
            #         selected_clients_grads.append(tmp_grad)
            #         local_model_param_cache.append(meta_train_param)
            # # client_id 对齐 selected_clients_grads
            # selected_clients = np.array(train_clients)

            clients_start_time = time.time()
            for client_id in selected_clients:
                # if client_id not in c_local.keys():
                #     c_local[client_id] = [torch.zeros_like(c) for c in c_global]
                # _, c_plus, c_delta, tmp_grad, meta_train_param = meta_client_process_scaffold(
                #     client_id, global_model, client_trainloaders, client_iter_trainloaders, c_global, c_local[client_id]
                # )
                _, tmp_grad, meta_train_param = meta_client_process(
                    client_id, global_model, client_trainloaders, client_iter_trainloaders
                )
                # c_local[client_id] = c_plus
                # c_delta_cache.append(c_delta)
                selected_clients_grads.append(tmp_grad)
                local_model_param_cache.append(meta_train_param)
            clients_end_time = time.time()
            logger.info("round {}: clients time is {}".format(current_round + 1, clients_end_time - clients_start_time))

            selected_clients_simila_matrix = create_similarity_matrix(selected_clients_grads)
            avg_simila_matrix = client_similarity_matrix[selected_clients[:, None], selected_clients] * 0.2 + selected_clients_simila_matrix * 0.8
            client_similarity_matrix[selected_clients[:, None], selected_clients] = avg_simila_matrix

            local_model_param_avg = Aggregators.fedavg_aggregate(local_model_param_cache)
            SerializationTool.deserialize_model(global_model, local_model_param_avg)

            # for c_g, c_d in zip(c_global, zip(*c_delta_cache)):
            #     c_d = torch.stack(c_d, dim=-1).sum(dim=-1)
            #     c_g.data += (1 / len(selected_clients)) * c_d.data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    config, cluster_partitioner, model = read_options()
    num_client = config['num_client']
    num_classes = config['num_classes']
    num_round = config['num_round']

    device = torch.device(get_best_gpu() if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")

    client_trainloaders, client_iter_trainloaders, client_testloaders = dataloader_init()

    client_sample_stream = [
        random.sample(
            range(num_client), max(1, int(num_client * 0.2))
        )
        for _ in range(num_round)
    ]

    # improve_test()
    # meta_cluster_test()
    meta_cluster_test2()


