import threading
from typing import List
from copy import deepcopy
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators

from fedavg import FedAvgServer

class AgglomerativeClusterFLServer(FedAvgServer):
    def __init__(self) -> None:
        super().__init__()

        self.clients_cluster: List[List[int]] = [[i] for i in range(len(self.clients))]
        self.cluster_model: List[torch.nn.Module] = [deepcopy(self.clients[i].local_model) for i in range(len(self.clients))]
        self.client_map_cluster_model = {i: self.cluster_model[i] for i in range(len(self.clients))}
        self.no_merge_count = 0
        self.similarity_matrix = np.full((len(self.clients), len(self.clients)), -1.0, dtype=np.float16)
        self.memory_matrix = np.full((len(self.clients), len(self.clients)), 0, dtype=np.int16)
    
    def update_matrix(self, grads_list: List[torch.Tensor], current_round: int):
        for i, grads_i in enumerate(grads_list):
            for j, grads_j in enumerate(grads_list[i + 1:], i + 1):
                if grads_i is None or grads_j is None:
                    continue
                similarity_score = torch.cosine_similarity(grads_i, grads_j, dim=0, eps=1e-12).item()
                self.similarity_matrix[i, j] = similarity_score
                self.similarity_matrix[j, i] = similarity_score

                self.memory_matrix[i, j] = current_round
                self.memory_matrix[j, i] = current_round

    def compute_matrics(self):
        a_max_min = -1.0
        cluster_1_index, cluster_2_index = -1, -1
        a_within_min = None
        a_cross_max = None
        a_cross_min = -1.0
        for i, cluster_i in enumerate(self.clients_cluster):
            for j, cluster_j in enumerate(self.clients_cluster[i + 1:], i + 1):
                a_min = np.min(np.where(
                    self.similarity_matrix[np.ix_(cluster_i, cluster_j)] > -1, 
                    self.similarity_matrix[np.ix_(cluster_i, cluster_j)], np.inf
                ))
                if np.isinf(a_min):
                    continue
                
                if a_min > a_max_min:
                    if len(cluster_i) > 1 and len(cluster_j) > 1:
                        mask1 = self.similarity_matrix[np.ix_(cluster_i, cluster_i)] > -1.0
                        mask2 = self.similarity_matrix[np.ix_(cluster_j, cluster_j)] > -1.0
                        min_cluster_i = np.min(self.similarity_matrix[np.ix_(cluster_i, cluster_i)][mask1]) if np.any(mask1) else np.inf
                        min_cluster_j = np.min(self.similarity_matrix[np.ix_(cluster_j, cluster_j)][mask2]) if np.any(mask2) else np.inf
                        if np.isinf(min_cluster_i) and np.isinf(min_cluster_j):
                            continue
                        a_within_min = min(min_cluster_i, min_cluster_j)
                    a_max_min = a_min
                    cluster_1_index, cluster_2_index = i, j
                    a_cross_max = np.max(self.similarity_matrix[np.ix_(cluster_i, cluster_j)])
                    a_cross_min = a_min
        return a_cross_min, a_within_min, a_cross_max, cluster_1_index, cluster_2_index


    def combine_clusters(self, cluster_1_index, cluster_2_index):
        self.logger.info(f"Combine cluster {cluster_1_index} and {cluster_2_index}")
        cluster_1 = self.clients_cluster[cluster_1_index]
        cluster_2 = self.clients_cluster[cluster_2_index]
        self.clients_cluster.remove(cluster_1)
        self.clients_cluster.remove(cluster_2)
        new_cluster = cluster_1 + cluster_2
        self.clients_cluster.append(new_cluster)

        cluster_1_model = self.cluster_model[cluster_1_index]
        cluster_2_model = self.cluster_model[cluster_2_index]
        self.cluster_model.remove(cluster_1_model)
        self.cluster_model.remove(cluster_2_model)
        new_model = deepcopy(self.global_model)
        self.cluster_model.append(new_model)

        for cid in new_cluster:
            self.client_map_cluster_model[cid] = new_model

    def aggregate_clusterwise(self, clients_cid: List[int]):
        for index, cluster in enumerate(self.clients_cluster):
            tmp_grads_list = [SerializationTool.serialize_model(self.clients[cid].local_model) for cid in cluster if cid in clients_cid]
            if len(tmp_grads_list) > 0:
                avg_param = Aggregators.fedavg_aggregate(tmp_grads_list)
                SerializationTool.deserialize_model(self.cluster_model[index], avg_param)

    def evaluate_clusterwise(self):
        loss_list, top1_list, top5_list = [], [], []
        for cluster in self.clients_cluster:
            results = [self.clients[cid].evaluate(self.client_map_cluster_model[cid]) for cid in cluster]
            loss_, top1_, top5_ = np.mean(list(zip(*results)), axis=1)
            if len(cluster) > 20:
                self.logger.info(f"{len(cluster)} clients in cluster evaluated, loss: {loss_}, top1: {top1_}, top5: {top5_}")
            else:
                self.logger.info(f"cluster {cluster} evaluated, loss: {loss_}, top1: {top1_}, top5: {top5_}")
            loss_list.append(loss_)
            top1_list.append(top1_)
            top5_list.append(top5_)
        return np.mean(loss_list), np.mean(top1_list), np.mean(top5_list)

    def run(self):
        eval_result = self.init_eval_result()
        for current_round in range(self.num_round):
            grads_list: List[torch.Tensor] = [None] * len(self.clients)
            selected_clients = self.clients_sample_stream[current_round]
            selected_clients_cid = [client.cid for client in selected_clients]
            self.logger.info(f"Round {current_round + 1} selected clients: {selected_clients_cid}")
            if self.no_merge_count <= 10:
                for client in selected_clients:
                    param_before = SerializationTool.serialize_model(self.global_model)
                    client.train(self.global_model)
                    param_after = SerializationTool.serialize_model(client.local_model)
                    grads_list[client.cid] = param_before - param_after
                
                self.update_matrix(grads_list, current_round)
                a_cross_min, a_within_min, a_cross_max, cluster_1_index, cluster_2_index = self.compute_matrics()
                self.logger.info(f"Round {current_round + 1} a_cross_min: {a_cross_min}, a_within_min: {a_within_min}, a_cross_max: {a_cross_max}")
                if a_cross_min > 0 and (a_within_min is None or a_cross_max > a_within_min):
                    self.combine_clusters(cluster_1_index, cluster_2_index)
                    self.no_merge_count = 0
                else:
                    self.no_merge_count += 1
                out_memory_indices =np.where((current_round - self.memory_matrix) > 10)
                self.similarity_matrix[out_memory_indices] = -1.0

                self.aggregate_clusterwise(selected_clients_cid)
                self.aggregate(selected_clients)
            
            else:
                for client in selected_clients:
                    client.train(self.client_map_cluster_model[client.cid])
                self.aggregate_clusterwise(selected_clients_cid)
        
            if (current_round + 1) % self.eval_interval == 0:
                loss, top1, top5 = self.evaluate_clusterwise()
                eval_result.loc[current_round + 1] = [loss, top1, top5]
                self.logger.info(f"Round {current_round + 1} evaluation result: loss: {loss}, top1: {top1}, top5: {top5}")
                thread = threading.Thread(target=self.save_eval_result, args=(eval_result,))
                thread.start()


if __name__ == "__main__":
    server = AgglomerativeClusterFLServer()
    server.run()