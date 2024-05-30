import threading
from typing import List
from copy import deepcopy
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators

from fedavg import FedAvgServer


class ClusterFLServer(FedAvgServer):
    def __init__(self) -> None:
        super().__init__()

        # 作者提供的参数
        self.eps_1 = 0.4
        self.eps_2 = 1.6

        # CIFAR10 - 2 Shards - LeNet
        # self.eps_1 = 0.07
        # self.eps_2 = 0.35

        # CIFAR100 - 20 Shards - LeNet
        # self.eps_1 = 0.1
        # self.eps_2 = 0.5

        # EMNIST - 10 Shards - LeNet
        # self.eps_1 = 0.16
        # self.eps_2 = 0.8

        self.clients_cluster: List[List[int]] = [list(range(len(self.clients)))]
        self.cluster_models: List[torch.nn.Module] = [deepcopy(self.global_model)]
        self.client_map_cluster = {i: 0 for i in range(len(self.clients))}
    
    def pairwise_similarity(self, grads_list: List[torch.Tensor]):
        similarity_matrix = np.eye(len(self.clients))
        for i, grads_i in enumerate(grads_list):
            for j, grads_j in enumerate(grads_list[i + 1:], i + 1):
                if grads_i is None or grads_j is None:
                    continue
                similarity_score = torch.cosine_similarity(grads_i, grads_j, dim=0, eps=1e-12).item()
                similarity_matrix[i, j] = similarity_score
                similarity_matrix[j, i] = similarity_score
        return similarity_matrix
    
    def aggregate_clusterwise(self, grads_list: List[torch.Tensor], clients_cid: List[int]):
        for idx, cluster in enumerate(self.clients_cluster):
            tmp_grads_list = [grads_list[cid] for cid in cluster if cid in clients_cid]
            if len(tmp_grads_list) == 0:
                continue
            cluster_model = self.cluster_models[idx]
            cluster_model.to('cpu')
            SerializationTool.deserialize_model(cluster_model, Aggregators.fedavg_aggregate(tmp_grads_list), mode='add')

    def evaluate_clusterwise(self):
        loss_list, top1_list, top5_list = [], [], []
        for idx, cluster in enumerate(self.clients_cluster):
            cluster_model = self.cluster_models[idx]
            results = [self.clients[cid].evaluate(cluster_model) for cid in cluster]
            loss_, top1_, top5_ = np.mean(list(zip(*results)), axis=1)
            if len(cluster) > 20:
                self.logger.info(f"{len(cluster)} clients in cluster evaluated, loss: {loss_:.4f}, top1: {top1_:.4f}, top5: {top5_:.4f}")
            else:
                self.logger.info(f"cluster {cluster} evaluated, loss: {loss_:.4f}, top1: {top1_:.4f}, top5: {top5_:.4f}")
            loss_list.append(loss_)
            top1_list.append(top1_)
            top5_list.append(top5_)
        loss, top1, top5 = np.mean(loss_list), np.mean(top1_list), np.mean(top5_list)
        return loss, top1, top5

    def run(self):
        eval_result = self.init_eval_result()
        grads_list = [None] * len(self.clients)
        global_max_norm = 0.0
        for current_round in range(self.num_round):
            selected_clients = self.clients_sample_stream[current_round]
            selected_clients_cid = np.array([client.cid for client in selected_clients])
            self.logger.info(f"Round {current_round + 1} selected clients: {selected_clients_cid}")
            
            for client in selected_clients:
                cluster_model = self.cluster_models[self.client_map_cluster[client.cid]]
                param_before = SerializationTool.serialize_model(cluster_model)
                client.train(cluster_model)
                param_after = SerializationTool.serialize_model(client.local_model)
                param_update = param_after - param_before
                grads_list[client.cid] = param_update

            similarity_matrix = self.pairwise_similarity(grads_list)

            if current_round > 20:
                clients_cluster_new = deepcopy(self.clients_cluster)
                for idx, cluster in enumerate(self.clients_cluster):
                    cluster = np.array(cluster)
                    grads_list_cluster = [grads_list[cid] for cid in cluster if grads_list[cid] is not None]
                    if len(grads_list_cluster) == 0:
                        continue
                    max_norm = max([torch.norm(grads).item() for grads in grads_list_cluster])
                    global_max_norm = max(global_max_norm, max_norm)
                    mean_norm = torch.norm(torch.mean(torch.stack(grads_list_cluster), dim=0)).item()
                    self.logger.info(f"Round {current_round + 1} mean norm {round(mean_norm, 4)}, max norm {round(max_norm, 4)}, global max norm {round(global_max_norm, 4)}")
                    if mean_norm < self.eps_1 and max_norm > self.eps_2 and len(cluster) > 2 and current_round > 20:
                        clustering = AgglomerativeClustering(
                            metric='precomputed', linkage='complete'
                        ).fit(-similarity_matrix[cluster][:, cluster])
                        cluster_1 = cluster[np.argwhere(clustering.labels_ == 0).flatten()]
                        cluster_2 = cluster[np.argwhere(clustering.labels_ == 1).flatten()]
                        self.logger.info(f"Cluster {cluster} split into {cluster_1} and {cluster_2}")
                        clients_cluster_new[idx] = cluster_1.tolist()
                        for cid in cluster_1:
                            self.client_map_cluster[cid] = idx
                        clients_cluster_new.append(cluster_2.tolist())
                        self.cluster_models.append(deepcopy(self.cluster_models[idx]))
                        for cid in cluster_2:
                            self.client_map_cluster[cid] = len(self.cluster_models) - 1
                self.clients_cluster = clients_cluster_new

            self.aggregate_clusterwise(grads_list, selected_clients_cid)

            if (current_round + 1) % self.eval_interval == 0:
                # evaluate clients in each cluster
                loss, top1, top5 = self.evaluate_clusterwise()
                eval_result.loc[current_round + 1] = [loss, top1, top5]
                self.logger.info(f"Round {current_round + 1} evaluation result: loss {loss:.4f}, top1 {top1:.4f}, top5 {top5:.4f}")
                thread = threading.Thread(target=self.save_eval_result, args=(eval_result,))
                thread.start()
            
                self.logger.info("global max norm: {:.4f}".format(global_max_norm))

if __name__ == "__main__":
    server = ClusterFLServer()
    server.run()

