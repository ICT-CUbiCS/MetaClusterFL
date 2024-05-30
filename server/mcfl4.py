from copy import deepcopy
import threading
import numpy as np
from sklearn.cluster import SpectralClustering
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

from mcfl import MetaClusterFLServer


class MetaClusterFLServer4(MetaClusterFLServer):
    def __init__(self) -> None:
        super().__init__()

        self.client_map_cluster = {i: 0 for i in range(len(self.clients))}

    def cluster_train(self, selected_clients_cid):
        
        # 保存当前轮次所有参与训练的客户端的模型参数, 同一聚类的客户端的模型参数放在一起
        cluster_model_param_cache = [[] for _ in range(len(self.cluster_clients))]
        for cid in selected_clients_cid:
            cluster_model_param_cache[self.client_map_cluster[cid]].append(
                SerializationTool.serialize_model(self.clients[cid].local_model))
        
        # 聚合放在一起的客户端模型参数, 得到聚类后的模型参数
        cluster_models = [None] * len(self.cluster_clients)
        for idx, params_cache in enumerate(cluster_model_param_cache):
            if len(params_cache) == 0:
                continue
            cluster_models[idx] = Aggregators.fedavg_aggregate(params_cache)
        
        # 更新聚类后的模型参数到每个客户端的 per_model
        # for client in self.clients:
        #     cluster_model = cluster_models[self.client_map_cluster[client.cid]]
        #     if cluster_model is not None:
        #         SerializationTool.deserialize_model(client.per_model, cluster_model)
        for cluster in self.cluster_clients:
            cluster_model = cluster_models[self.client_map_cluster[cluster[0]]]
            if cluster_model is not None:
                SerializationTool.deserialize_model(self.clients[cluster[0]].per_model, cluster_model)
        

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
            self.client_map_cluster[index] = label
        new_cluster_clients = list(cluster_dict.values())
        # new_cluster_clients = self._sort_cluster_clients(list(cluster_dict.values()))
        return new_cluster_clients

    def _split_cluster(self, cluster_index, clients_cluster_1, clients_cluster_2):
        self.logger.info(f"Cluster {self.cluster_clients[cluster_index]} is split into {clients_cluster_1} and {clients_cluster_2}")
        self.cluster_clients[cluster_index] = clients_cluster_1.tolist()
        self.cluster_clients.append(clients_cluster_2.tolist())
        assigned_cluster_label = set(self.client_map_cluster.values())
        all_cluster_label = set(range(len(self.cluster_clients)))
        unassigned_cluster_label = (all_cluster_label - assigned_cluster_label).pop()
        for cid in self.cluster_clients[-1]:
            self.client_map_cluster[cid] = unassigned_cluster_label
        self.inital_num_clusters += 1

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
                    if self.inital_num_clusters > 2:
                        self.inital_num_clusters -= 1

    def evaluate_clusterwise(self, current_round):
        loss_list, top1_list, top5_list = [], [], []
        for cluster in self.cluster_clients:
            results = []
            for cid in cluster:
                if self.accumulated_inactive <= 10:
                    results.append(self.clients[cid].evaluate(self.global_model))
                else:
                    results.append(self.clients[cid].evaluate(self.clients[cid].per_model))
            loss_, top1_, top5_ = np.mean(list(zip(*results)), axis=1)
            self.logger.info(f"Round {current_round}: {cluster} loss: {loss_}, top1: {top1_}, top5: {top5_}")
            loss_list.append(loss_)
            top1_list.append(top1_)
            top5_list.append(top5_)
        return np.mean(loss_list), np.mean(top1_list), np.mean(top5_list)

    def run(self):
        eval_result = self.init_eval_result()
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
                
                # 使用 全局元模型 初始化客户端的 个性化模型
                if self.accumulated_inactive == 10:
                    for cluster in self.cluster_clients:
                        cluster_model = deepcopy(self.global_model)
                        for cid in cluster:
                            self.clients[cid].per_model = cluster_model
                    # for client in self.clients:
                    #     client.per_model.load_state_dict(self.global_model.state_dict())
                    self.cluster_train(selected_clients_cid)
                    self.accumulated_inactive += 1
                
            else:
                # 继续联邦元学习
                for client in selected_clients:
                    client.train(client.per_model)
                
                # 聚类内 聚合模型参数, 以更新客户端的 个性化模型
                self.cluster_train(selected_clients_cid)

            if (current_round + 1) % self.eval_interval == 0:
                if self.accumulated_inactive < 10:
                    # 相似度热力图
                    thread = threading.Thread(target=self.plot_hot_map, args=(current_round,))
                    thread.start()

                    # 聚类
                    count_zero_in_similarity_matrix = self.similarity_matrix.size - np.count_nonzero(self.similarity_matrix)
                    self.logger.info(f"{count_zero_in_similarity_matrix} zero in similarity matrix.")
                    if count_zero_in_similarity_matrix == len(self.clients):
                        num_clusters = len(self.cluster_clients)
                        # 谱聚类
                        new_cluster_clients = self.spectral_clustering(self.inital_num_clusters)

                        # 检查聚类
                        self.cluster_clients = new_cluster_clients
                        self.check_cluster()

                        if num_clusters == len(self.cluster_clients):
                            self.accumulated_inactive += 1

                        # 计算当前聚类切图权重
                        cut_weight = self._compute_cut_weight()
                        self.logger.info(f"Round {current_round + 1} cut weight: {cut_weight}")
                        self.logger.info(f"Round {current_round + 1} cluster_num: {len(self.cluster_clients)}")

                # 评估
                loss, top1, top5 = self.evaluate_clusterwise(current_round)
                eval_result.loc[current_round + 1] = [loss, top1, top5]
                self.logger.info(f"Round {current_round + 1} evaluation result: loss {loss:.4f}, top1 {top1:.4f}, top5 {top5:.4f}")
                thread = threading.Thread(target=self.save_eval_result, args=(eval_result,))
                thread.start()


if __name__ == "__main__":
    server = MetaClusterFLServer4()
    server.run()