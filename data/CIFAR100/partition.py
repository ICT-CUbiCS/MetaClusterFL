import os
import random
import json
from typing import Dict
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from fedlab.contrib.dataset.basic_dataset import FedDataset, BaseDataset
from fedlab.utils.dataset.partition import CIFAR100Partitioner

class Partition(FedDataset):
    
    def __init__(self, **kwargs) -> None:
        
        # 原始数据存放目录
        self.root = os.path.join(os.path.dirname(__file__), "raw")
        
        # 数据划分存放目录 
        self.path = os.path.join(
            os.path.dirname(__file__), 
            "partition", 
            "{}_{}".format(kwargs['balance'], kwargs['partition'])
        )
        
        self.num_classes = 100  # 固定值, 取决于数据集本身
        self.trainset_rate = 0.8  # 训练集比例
        self.max_samples_per_class = 50 # 每个客户端最大样本数
        
        # 其他参数
        self.num_client = kwargs['num_client']
        self.num_cluster = kwargs['num_cluster']
        self.num_client_per_cluster = self.num_client // self.num_cluster

        self.img_mean = (0.5071, 0.4867, 0.4408)
        self.img_std = (0.2675, 0.2565, 0.2761)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(os.path.join(self.path, "train"))
            os.makedirs(os.path.join(self.path, "test"))
            trainset, testset, cluster_indices = self.setup_partition(**kwargs)
            self.preprocess(trainset, testset, cluster_indices)
        else:
            # 加载 testset
            # testset = torch.load(os.path.join(self.path, "testset.pkl"))
            # self.testset = DataLoader(testset, batch_size=128, shuffle=True)
            
            with open(os.path.join(self.path, "partition_stat.json"), "r") as f:
                self.partition_stat = json.load(f)


    def setup_partition(self, **kwargs):
        trainset = CIFAR100(root=self.root, train=True, download=True)
        testset = CIFAR100(root=self.root, train=False, download=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        trainset.transform = transform
        testset.transform = transform

        cluster_partitioner = CIFAR100Partitioner(
            targets=trainset.targets, num_clients=self.num_cluster,
            balance=kwargs['balance'], partition=kwargs['partition'],
            unbalance_sgm=kwargs['unbalance_sgm'], num_shards=kwargs['num_shards'],
            dir_alpha=kwargs['dir_alpha'], verbose=kwargs['verbose'],
            min_require_size=kwargs['min_require_size'], seed=kwargs['seed']
        )

        return trainset, testset, cluster_partitioner.client_dict


    def preprocess(self, trainset, testset, cluster_indices: Dict[int, np.ndarray]):

        testset_target = np.array(testset.targets)
        test_label_idx = [np.where(testset_target == i)[0] for i in range(self.num_classes)]
        trainset_target = np.array(trainset.targets)
        train_label_idx = [np.where(trainset_target == i)[0] for i in range(self.num_classes)]

        # 统计 cluster 划分后的数据
        clusters_stat = {}
        
        # 统计 client 划分后的数据
        clients_stat = {}
        
        # 存储 cluster 划分后的数据
        allocate_testset = BaseDataset([], [])
        for cluster_id, indices in cluster_indices.items():
            cluster_stat, train_labels_dict, test_labels_dict, cluster_testset = self._assign_cluster_data(cluster_id, indices, trainset, train_label_idx, testset, test_label_idx)
            clusters_stat[str(cluster_id)] = cluster_stat
            
            cluster_clients_stat = self._assign_client_data(cluster_id, train_labels_dict, trainset, test_labels_dict, testset)
            clients_stat.update(cluster_clients_stat)
            
            allocate_testset += cluster_testset
        
        # 存储 testset
        torch.save(allocate_testset, os.path.join(self.path, "testset.pkl"))
        # self.testset = DataLoader(testset, batch_size=128, shuffle=True)
        
        self.partition_stat = {
            "clusters": clusters_stat,
            "clients": clients_stat
        }
        with open(os.path.join(self.path, "partition_stat.json"), "w") as f:
            json.dump(self.partition_stat, f, indent=4)


    def _assign_cluster_data(self, cluster_id, cluster_indices, trainset, train_label_idx, testset, test_label_idx):
        """按照划分为 cluster 分配具体数据"""

        # 记录 cluster 每种 label 在 trainset 中的 indices
        train_labels_dict = defaultdict(list)
        for idx in cluster_indices:
            img, label = trainset[idx]
            train_labels_dict[label].append(idx)
        
        num_samples_per_cluster = self.num_client_per_cluster * len(train_labels_dict.keys()) * self.max_samples_per_class
        num_samples_rate_cluster = float(num_samples_per_cluster) / float(len(cluster_indices))
        for key in train_labels_dict.keys():
            train_labels_dict[key] = random.sample(
                train_labels_dict[key], 
                min(len(train_labels_dict[key]), int(num_samples_rate_cluster * len(train_labels_dict[key])))
            )
        # train_images = [trainset[idx][0] for indices in train_labels_dict.values() for idx in indices]
        # train_labels = [trainset[idx][1] for indices in train_labels_dict.values() for idx in indices]
        train_images, train_labels = zip(*[(trainset[idx][0], trainset[idx][1]) for indices in train_labels_dict.values() for idx in indices])
        
        # 记录 cluster 每种 label 在 testset 中的 indices
        test_labels_dict = defaultdict(list)
        test_images, test_labels = [], []
        for label, train_indices in train_labels_dict.items():
            label_rate = float(len(train_indices)) / float(len(train_label_idx[label]))
            test_indices = random.sample(list(test_label_idx[label]), int(label_rate * len(test_label_idx[label])))
            test_labels_dict[label] = test_indices
            # test_images.extend([testset[idx][0] for idx in test_indices])
            # test_labels.extend([testset[idx][1] for idx in test_indices])
            for idx in test_indices:
                img, label = testset[idx]
                test_images.append(img)
                test_labels.append(label)

        # 构建 cluster_trainset 和 cluster_testset
        cluster_trainset = BaseDataset(train_images, train_labels)
        cluster_testset = BaseDataset(test_images, test_labels)
        # 存储 cluster 划分后的数据
        torch.save(cluster_trainset, os.path.join(self.path, "train", "cluster_{}.pkl".format(cluster_id)))
        torch.save(cluster_testset, os.path.join(self.path, "test", "cluster_{}.pkl".format(cluster_id)))
        # 统计 cluster 划分后的 label 种类及数量
        train_label_count = Counter(cluster_trainset.y)
        test_label_count = Counter(cluster_testset.y)
        cluster_stat = {
            "train": dict(train_label_count),
            "test": dict(test_label_count)
        }
        
        return cluster_stat, train_labels_dict, test_labels_dict, cluster_testset


    def _assign_client_data(self, cluster_id, train_labels_dict, trainset, test_labels_dict, testset):
        """划分 cluster 中的 clients 数据
        """
        clients_stat = {}
        for i in range(self.num_client_per_cluster):
            client_id = cluster_id * self.num_client_per_cluster + i
            # 分配 trainset
            train_images, train_labels = [], []
            for label, indices in train_labels_dict.items():
                num_samples = len(indices) // self.num_client_per_cluster
                client_train_indices = indices[i * num_samples: (i + 1) * num_samples]
                # client_train_indices = random.sample(indices, len(indices) // self.num_client_per_cluster)
                # client_train_indices = random.sample(indices, int(0.8 * len(indices)))
                for idx in client_train_indices:
                    img, label = trainset[idx]
                    train_images.append(img)
                    train_labels.append(label)

            # 分配 testset
            test_images, test_labels = [], []
            for label, indices in test_labels_dict.items():
                num_samples = len(indices) // self.num_client_per_cluster
                client_test_indices = indices[i * num_samples: (i + 1) * num_samples]
                for idx in client_test_indices:
                    img, label = testset[idx]
                    test_images.append(img)
                    test_labels.append(label)

            client_trainset = BaseDataset(train_images, train_labels)
            client_testset = BaseDataset(test_images, test_labels)
            # 存储 client 划分后的数据
            torch.save(client_trainset, os.path.join(self.path, "train", "client_{}.pkl".format(client_id)))
            torch.save(client_testset, os.path.join(self.path, "test", "client_{}.pkl".format(client_id)))
            # 统计 client 划分后的 label 种类及数量
            train_label_count = Counter(client_trainset.y)
            test_label_count = Counter(client_testset.y)
            clients_stat[str(client_id)] = {
                "train": dict(train_label_count),
                "test": dict(test_label_count),
                "cluster": cluster_id,
            }
        
        return clients_stat


    def _subset_dataset(self, dataset, indices):
        """数据抽取
            从数据集 dataset 中抽取对应索引 indices 的数据
        """
        images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        
        return BaseDataset(images, labels)


    def _mean_std_compute(self, trainset, testset):
        """计算数据集的均值和标准差
            分别计算 transet 和 testset 中 images 三通道的平均值和标准差, 再取平均
        """
        
        train_mean = trainset.data.mean(axis=(0,1,2)) / 255
        train_std = trainset.data.std(axis=(0,1,2)) / 255

        test_mean = testset.data.mean(axis=(0,1,2)) / 255
        test_std = testset.data.std(axis=(0,1,2)) / 255

        avg_mean = (train_mean + test_mean) / 2
        avg_std = (train_std + test_std) / 2

        return tuple(avg_mean.round(4)), tuple(avg_std.round(4))


    def get_dataset(self, cid, type="train", cluster=False):
        """获取单个 client 的数据集
        
        Args:
            cid: client id
            type: train or test
            cluster: 是否获取 client 对应的 cluster 数据集
        
        Returns:
            dataset: torch.utils.data.Dataset
        """
        if cluster:
            cluster_id = self.partition_stat["clients"][str(cid)]["cluster"]
            dataset = torch.load(os.path.join(self.path, type, "cluster_{}.pkl".format(cluster_id)))
        else:
            dataset = torch.load(os.path.join(self.path, type, "client_{}.pkl".format(cid)))
        
        return dataset


    def get_dataloader(self, cid, batch_size, type="train", cluster=False):
        dataset = self.get_dataset(cid, type, cluster)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader


    def get_cluster_dataloader(self, clients_id, batch_size, type="train"):
        """获取 cluster 的 dataloader
        """
        if len(clients_id) == 1:
            return self.get_dataloader(clients_id[0], batch_size, type, cluster=True)
        # 根据 clients_id 获取 不同的 cluster_id
        clusters_clients_id = {}
        for cid in clients_id:
            cluster_id = self.partition_stat["clients"][str(cid)]["cluster"]
            cluster_clients_id = clusters_clients_id.get(cluster_id, [])
            cluster_clients_id.append(cid)
            clusters_clients_id[cluster_id] = cluster_clients_id
        
        # 合并 cluster 的 dataset
        clusters_dataset = BaseDataset([], [])
        for cluster_id, cluster_clients_id in clusters_clients_id.items():
            cluster_dataset = self.get_dataset(cluster_clients_id[0], type, cluster=True)
            clusters_dataset += cluster_dataset
        
        # 获取 dataloader
        batch_size = min(batch_size, len(clusters_dataset))
        dataloader = DataLoader(clusters_dataset, batch_size=batch_size, shuffle=True)
        return dataloader


    def plot_clients_labels(self):
        """绘制 num_cluster 个 属于不同 cluster 的 client label 分布图
        """
        columns = [int(i * self.num_client_per_cluster + random.randint(0, self.num_client_per_cluster-1)) for i in range(self.num_cluster)]
        rows = [i for i in range(self.num_classes)]
        df = pd.DataFrame(columns=columns, index=rows, data=0)
        for cid in columns:
            client_stat = self.partition_stat["clients"][str(cid)]
            for label, count in client_stat["train"].items():
                df.loc[int(label), int(cid)] = count
        
        df.plot(kind='bar', stacked=True, figsize=(20, 10))
        plt.legend(loc='upper right')
        plt.xlabel("label")
        plt.ylabel("count")
        plt.title("Clients Labels Distribution")
        plt.tight_layout()
        plt_dir = os.path.join(self.path, "plot")
        if not os.path.exists(plt_dir):
            os.makedirs(plt_dir)
        plt.savefig(os.path.join(plt_dir, "clients_labels_distribution.png"), dpi=600)

    def plot_clusters_labels(self):
        """绘制 cluster 的 label 分布图
            堆叠图, 横坐标为 label, 纵坐标为 count, 每个 cluster 在对应的 label 上的 count 堆叠
            利用 pandas 的 DataFrame.plot(kind='bar', stacked=True) 绘制, 第一维需要为 label, 第二维为 cluster
            
            图片保存在 self.path/plot/cluster_labels_distribution.png
        """
        columns = [i for i in range(self.num_cluster)]
        rows = [i for i in range(self.num_classes)]
        df = pd.DataFrame(columns=columns, index=rows, data=0)
        for cluster_id, cluster_stat in self.partition_stat["clusters"].items():
            for label, count in cluster_stat["train"].items():
                df.loc[int(label), int(cluster_id)] = count
        
        df.plot(kind='bar', stacked=True, figsize=(20, 10))
        plt.legend(loc='upper right')
        plt.xlabel("labels")
        plt.ylabel("count")
        plt.title("cluster labels distribution")
        plt.tight_layout()
        plot_dir = os.path.join(self.path, "plot")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, "cluster_labels_distribution.png"), dpi=600)


    def plot_random_cluster(self):
        """绘制随机 cluster 及其 client 的 label 分布图
            堆叠图, 横坐标为 label, 纵坐标为 count, 每个 client 在对应的 label 上的 count 堆叠
            利用 pandas 的 DataFrame.plot(kind='bar', stacked=True) 绘制, 第一维需要为 label, 第二维为 client
            
            图片保存在 self.path/plot/cluster_{random_cluster_id}_labels_distribution.png
        """
        cluster_id = random.randint(0, self.num_cluster-1)
        cluster_stat = self.partition_stat["clusters"][str(cluster_id)]
        columns = [self.num_client_per_cluster * cluster_id + i for i in range(self.num_client_per_cluster)]
        rows = sorted([int(label) for label in cluster_stat["train"].keys()])
        trainset_df = pd.DataFrame(columns=[-1] + columns, index=rows, data=0)
        testset_df = pd.DataFrame(columns=[-1] + columns, index=rows, data=0)
        
        # 统计 cluster 的 label 数据
        for label, count in cluster_stat["train"].items():
            trainset_df.loc[int(label), -1] = count
        for label, count in cluster_stat["test"].items():
            testset_df.loc[int(label), -1] = count
        
        # 统计 client 的 label 数据
        for client_id in columns:
            client_stat = self.partition_stat["clients"][str(client_id)]
            for label, count in client_stat["train"].items():
                trainset_df.loc[int(label), int(client_id)] = count
            
            for label, count in client_stat["test"].items():
                testset_df.loc[int(label), int(client_id)] = count
            
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        
        # cluster trainset 子图
        trainset_df[[-1]].plot(kind='bar', ax=ax1, figsize=(20, 10))
        ax1.set_title("cluster {} train labels distribution".format(cluster_id))
        
        # cluster testset 子图
        testset_df[[-1]].plot(kind='bar', ax=ax2, figsize=(20, 10))
        ax2.set_title("cluster {} test labels distribution".format(cluster_id))
        
        # clients trainset 子图
        trainset_df[columns].plot(kind='bar', stacked=True, ax=ax3, figsize=(20, 10))
        ax3.set_title("cluster {} clients train labels distribution".format(cluster_id))
        
        # clients testset 子图
        testset_df[columns].plot(kind='bar', stacked=True, ax=ax4, figsize=(20, 10))
        ax4.set_title("cluster {} clients test labels distribution".format(cluster_id))
        
        # 设置 axis
        for ax in (ax1, ax2, ax3, ax4):
            ax.legend(loc='upper right')
            ax.set_xlabel('labels')
            ax.set_ylabel('count')
            if len(rows) > 20:
                ax.set_xticks(np.arange(0, len(rows), step=5))
        
        plt.tight_layout()
        plot_dir = os.path.join(self.path, "plot")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, "cluster_{}_labels_distribution.png".format(cluster_id)), dpi=600)


def construct_partition():
    """构造Partition 类
        构造类的初始化参数和 project_dir/config.json 同步
    """
    
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(project_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    partition_init_args = {
        "num_client": config["num_client"],
        "num_cluster": config["num_cluster"],
        
        "balance": config["balance"],
        "partition": config["partition"],
        "unbalance_sgm": config["unbalance_sgm"],
        "num_shards": config["num_shards"],
        "dir_alpha": config["dir_alpha"],
        "verbose": config["verbose"],
        "min_require_size": config["min_require_size"],
        "seed": config["seed"],
        
        # "accelerate": config["accelerate"]
    }
    
    partitioner = Partition(**partition_init_args)
    
    return partitioner


if __name__ == "__main__":
    
    cifar10_partition = construct_partition()
    cifar10_partition.plot_clusters_labels()
    cifar10_partition.plot_random_cluster()
    cifar10_partition.plot_clients_labels()