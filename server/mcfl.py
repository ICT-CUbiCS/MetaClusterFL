import os
import threading
from copy import deepcopy
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

from perfedavg import PerFedAvgServer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MetaClusterFLServer(PerFedAvgServer):
    def __init__(self) -> None:
        super().__init__()

        self.similarity_matrix = np.full((len(self.clients), len(self.clients)), 0, dtype=np.float16)
        self.cluster_clients: List[List[int]] = [[i for i in range(len(self.clients))]]
        self.inital_num_clusters = 4
        self.tradeoff_factor = 0.8
        self.accumulated_inactive = 0

    def pairwise_similarity(self, grads_list: List[torch.Tensor]):
        for i, grads_i in enumerate(grads_list):
            for j, grads_j in enumerate(grads_list[i + 1:], i + 1):
                if grads_i is None or grads_j is None:
                    continue
                similarity_score = torch.cosine_similarity(grads_i, grads_j, dim=0, eps=1e-12).item()
                self.similarity_matrix[i, j] = self.similarity_matrix[i, j] * (1 - self.tradeoff_factor) + similarity_score * self.tradeoff_factor
                self.similarity_matrix[j, i] = self.similarity_matrix[i, j]

    def seek_within_min_similarity(self, clients_cluster_1, clients_cluster_2):
        mask1 = np.eye(len(clients_cluster_1), dtype=bool)
        mask2 = np.eye(len(clients_cluster_2), dtype=bool)
        masked_cluster1_matrix = self.similarity_matrix[np.ix_(clients_cluster_1, clients_cluster_1)][~mask1]
        masked_cluster2_matrix = self.similarity_matrix[np.ix_(clients_cluster_2, clients_cluster_2)][~mask2]
        if masked_cluster1_matrix.size > 0 and masked_cluster2_matrix.size > 0:
            return min(np.min(masked_cluster1_matrix), np.min(masked_cluster2_matrix))
        elif masked_cluster1_matrix.size > 0:
            return np.min(masked_cluster1_matrix)
        elif masked_cluster2_matrix.size > 0:
            return np.min(masked_cluster2_matrix)
        else:
            return 0

    def _compute_metrics(self):
        global_min = 1.0
        cluster_index = -1
        clients_cluster_1 = None
        clients_cluster_2 = None
        for index, cluster in enumerate(self.cluster_clients):
            if len(cluster) <= 2:
                continue
            clients = np.array(cluster)
            clients_similarity_matrix = self.similarity_matrix[np.ix_(clients, clients)]
            # 去除待计算相似度矩阵中的小于0的值
            computed_clients_similarity_matrix = np.where(clients_similarity_matrix < 0, 0, clients_similarity_matrix)
            clients_clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
            clients_clustering.fit(computed_clients_similarity_matrix)
            clients_cluster_index_1 = np.argwhere(clients_clustering.labels_ == 0).reshape(-1)
            clients_cluster_index_2 = np.argwhere(clients_clustering.labels_ == 1).reshape(-1)
            a_min = np.min(np.where(
                clients_similarity_matrix[np.ix_(clients_cluster_index_1, clients_cluster_index_2)] != 0,
                clients_similarity_matrix[np.ix_(clients_cluster_index_1, clients_cluster_index_2)], np.inf
            ))
            if np.isinf(a_min):
                continue
            if a_min > 0 and (len(clients_cluster_index_1) == 1 or len(clients_cluster_index_2) == 1):
                continue
            if a_min < global_min:
                global_min = a_min
                cluster_index = index
                clients_cluster_1 = clients[clients_cluster_index_1]
                clients_cluster_2 = clients[clients_cluster_index_2]
        return global_min, cluster_index, clients_cluster_1, clients_cluster_2

    def _split_cluster(self, cluster_index, clients_cluster_1, clients_cluster_2):
        self.logger.info(f"Cluster {self.cluster_clients[cluster_index]} is split into {clients_cluster_1} and {clients_cluster_2}")
        self.cluster_clients[cluster_index] = clients_cluster_1.tolist()
        self.cluster_clients.append(clients_cluster_2.tolist())
        self.inital_num_clusters += 1
        self.accumulated_inactive = 0

    def check_cluster(self):
        global_min, cluster_index, clients_cluster_1, clients_cluster_2 = self._compute_metrics()
        if cluster_index != -1:
            self.logger.info(f"global_min: {global_min} between {clients_cluster_1} and {clients_cluster_2}")
            if global_min < 0:
                self._split_cluster(cluster_index, clients_cluster_1, clients_cluster_2)
            else:
                a_cross_max = np.max(self.similarity_matrix[np.ix_(clients_cluster_1, clients_cluster_2)])
                a_within_min = self.seek_within_min_similarity(clients_cluster_1, clients_cluster_2)
                self.logger.info(f"a_cross_max: {a_cross_max}, a_within_min: {a_within_min} between {clients_cluster_1} and {clients_cluster_2}")
                if a_cross_max <= a_within_min:
                    self._split_cluster(cluster_index, clients_cluster_1, clients_cluster_2)
                else:
                    self.accumulated_inactive += 1
                    self.logger.info(f"Cluster is not split in continuous {self.accumulated_inactive} rounds")

    def _compute_cut_weight(self):
        cut_weight = 0
        for cluster in self.cluster_clients:
            mask = np.zeros(self.similarity_matrix[0].shape, dtype=bool)
            mask[cluster] = True
            cluster_W = np.sum(self.similarity_matrix[mask, :][:, ~mask])
            temp_matrix = np.where(self.similarity_matrix[mask, :] < 0 , 0, self.similarity_matrix[mask, :])
            cluster_vol = np.sum(temp_matrix)
            cut_weight += cluster_W / cluster_vol
        return cut_weight

    def evaluate_clusterwise(self, current_round: int):
        loss_list, top1_list, top5_list = [], [], []
        for cluser_id, cluster in enumerate(self.cluster_clients):
            results = []
            for cid in cluster:
                if self.accumulated_inactive <= 10:
                    results.append(self.clients[cid].evaluate(self.global_model))
                else:
                    results.append(self.clients[cid].evaluate(self.cluster_model[cluser_id]))
            loss_, top1_, top5_ = np.mean(list(zip(*results)), axis=1)
            self.logger.info(f"Round {current_round}: {cluster} loss: {loss_}, top1: {top1_}, top5: {top5_}")
            loss_list.append(loss_)
            top1_list.append(top1_)
            top5_list.append(top5_)
        return np.mean(loss_list), np.mean(top1_list), np.mean(top5_list)

    def plot_hot_map(self, current_round: int):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(self.similarity_matrix , cmap='hot', interpolation='nearest')
        ax.set_title("round {} similarity hot map".format(current_round + 1))
        fig.tight_layout()
        img_dir = os.path.join(PROJECT_DIR, "result", "plot", self.__class__.__name__)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_path = os.path.join(img_dir, "round_{0:0>3}.png".format(current_round + 1))
        plt.savefig(img_path, dpi=300)
        plt.close()

        # save similarity matrix
        similarity_dataframe = pd.DataFrame(self.similarity_matrix)
        similarity_dataframe.to_csv(os.path.join(img_dir, "round_{0:0>3}.csv".format(current_round + 1)))

    def spectral_clustering(self, num_clusters: int):
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
        # 去除待计算相似度矩阵中的小于0的值
        computed_similarity_matrix = np.where(self.similarity_matrix < 0, 0, self.similarity_matrix)
        clustering.fit(computed_similarity_matrix)
        cluster_dict = {}
        for index, label in enumerate(clustering.labels_):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(index)
        new_cluster_clients = list(cluster_dict.values())
        return new_cluster_clients

    def init_cut_weight_result(self):
        columns = ["round", "cut_weight"]
        cut_weight_result = pd.DataFrame(columns=columns)
        cut_weight_result.set_index("round", inplace=True)
        return cut_weight_result
    
    def save_cut_weight_result(self, cut_weight_result: pd.DataFrame):
        result_dir = os.path.join(PROJECT_DIR, 'result', 'evaluation', 'server')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'cut_weight_in_{}.csv'.format(self.__class__.__name__))
        cut_weight_result.to_csv(result_path)

    def run(self):
        eval_result = self.init_eval_result()
        cut_weight_result = self.init_cut_weight_result()
        for current_round in range(self.num_round):
            selected_clients = self.clients_sample_stream[current_round]
            selected_clients_cid = [client.cid for client in selected_clients]
            self.logger.info(f"Round {current_round + 1} selected clients: {selected_clients_cid}")
                
            if self.accumulated_inactive <= 10:
                selected_clients_grads = [None] * len(self.clients)
                selected_clients_model_cache = []
                for client in selected_clients:
                    # 计算一步训练前后的模型参数差, 用于后续的相似度计算
                    client.local_model.load_state_dict(self.global_model.state_dict())
                    tmp_param_before = SerializationTool.serialize_model(client.local_model)
                    client.train_once()
                    tmp_param_after = SerializationTool.serialize_model(client.local_model)
                    tmp_param_update = tmp_param_before - tmp_param_after
                    selected_clients_grads[client.cid] = tmp_param_update
                    # 常规元学习
                    client.train(self.global_model)
                    selected_clients_model_cache.append(SerializationTool.serialize_model(client.local_model))
                
                # 更新相似度矩阵
                self.pairwise_similarity(selected_clients_grads)

                # 更新元模型
                avg_param = Aggregators.fedavg_aggregate(selected_clients_model_cache)
                SerializationTool.deserialize_model(self.global_model, avg_param)

                # 固定聚类
                if self.accumulated_inactive == 10:
                    # 计算聚类内元模型
                    self.cluster_model = [deepcopy(self.global_model) for _ in range(len(self.cluster_clients))]
                    for cluster_id, cluster in enumerate(self.cluster_clients):
                        cluster_model_param_cache = []
                        for cid in cluster:
                            if cid in selected_clients_cid:
                                cluster_model_param_cache.append(SerializationTool.serialize_model(self.clients[cid].local_model))
                        if len(cluster_model_param_cache) == 0:
                            continue
                        cluster_avg_param = Aggregators.fedavg_aggregate(cluster_model_param_cache)
                        SerializationTool.deserialize_model(self.cluster_model[cluster_id], cluster_avg_param)
                    self.accumulated_inactive += 1

                if (current_round + 1) % self.eval_interval == 0:
                    # 相似度热力图
                    thread = threading.Thread(target=self.plot_hot_map, args=(current_round,))
                    thread.start()

                    count_zero_in_similarity_matrix = self.similarity_matrix.size - np.count_nonzero(self.similarity_matrix)
                    self.logger.info(f"{count_zero_in_similarity_matrix} zero in similarity matrix.")
                    if count_zero_in_similarity_matrix == len(self.clients) and self.accumulated_inactive < 10 :
                        # 聚类
                        self.cluster_clients = self.spectral_clustering(self.inital_num_clusters)

                        # 检查聚类
                        self.check_cluster()

                        # 计算当前聚类切图权重
                        cut_weight = self._compute_cut_weight()
                        self.logger.info(f"Round {current_round + 1} cut weight: {cut_weight}")
                        self.logger.info(f"Round {current_round + 1} cluster_num: {len(self.cluster_clients)}")
                        cut_weight_result.loc[current_round + 1] = [cut_weight]
                        thread = threading.Thread(target=self.save_cut_weight_result, args=(cut_weight_result,))
                        thread.start()

                    # 评估
                    loss, top1, top5 = self.evaluate_clusterwise(current_round)
                    eval_result.loc[current_round + 1] = [loss, top1, top5]
                    self.logger.info(f"Round {current_round + 1} evaluation result: loss {loss:.4f}, top1 {top1:.4f}, top5 {top5:.4f}")
                    thread = threading.Thread(target=self.save_eval_result, args=(eval_result,))
                    thread.start()
            else:
                for cluster_id, cluster in enumerate(self.cluster_clients):
                    cluster_model_param_cache = []
                    for cid in cluster:
                        if cid in selected_clients_cid:
                            self.clients[cid].train(self.cluster_model[cluster_id])
                            cluster_model_param_cache.append(SerializationTool.serialize_model(self.clients[cid].local_model))
                    if len(cluster_model_param_cache) == 0:
                        continue
                    cluster_avg_param = Aggregators.fedavg_aggregate(cluster_model_param_cache)
                    SerializationTool.deserialize_model(self.cluster_model[cluster_id], cluster_avg_param)
                
                if (current_round + 1) % self.eval_interval == 0:
                    # 评估
                    loss, top1, top5 = self.evaluate_clusterwise(current_round)
                    eval_result.loc[current_round + 1] = [loss, top1, top5]
                    self.logger.info(f"Round {current_round + 1} evaluation result: loss {loss:.4f}, top1 {top1:.4f}, top5 {top5:.4f}")
                    thread = threading.Thread(target=self.save_eval_result, args=(eval_result,))
                    thread.start()

if __name__ == '__main__':
    server = MetaClusterFLServer()
    server.run()