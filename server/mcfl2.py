from copy import deepcopy
import numpy as np
import threading
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

from mcfl import MetaClusterFLServer


class MetaClusterFLServer2(MetaClusterFLServer):
    def __init__(self) -> None:
        super().__init__()

    def evaluate_clusterwise(self, current_round: int):
        loss_list, top1_list, top5_list = [], [], []
        for cluser_id, cluster in enumerate(self.cluster_clients):
            results = []
            for cid in cluster:
                if self.accumulated_inactive <= 10:
                    results.append(self.clients[cid].evaluate(self.global_model))
                else:
                    results.append(self.clients[cid].evaluate_without_train(self.cluster_model[cluser_id]))
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
                            self.clients[cid].train_after_cluster(self.cluster_model[cluster_id])
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
    server = MetaClusterFLServer2()
    server.run()