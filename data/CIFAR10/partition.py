import sys
import os
import json
from typing import Dict
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from fedlab.utils.dataset.partition import CIFAR10Partitioner

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)
from data.CIFAR100.partition import Partition as CIFAR100Partition

class Partition(CIFAR100Partition):
    
    def __init__(self, **kwargs) -> None:
        
        # 原始数据存放目录
        self.root = os.path.join(os.path.dirname(__file__), "raw")
        
        # 数据划分存放目录 
        self.path = os.path.join(
            os.path.dirname(__file__), 
            "partition", 
            "{}_{}".format(kwargs['balance'], kwargs['partition'])
        )
        
        self.num_classes = 10  # 固定值, 取决于数据集本身
        self.trainset_rate = 0.8  # 训练集比例
        self.max_samples_per_class = 50 # 每个客户端每种label最大样本数
        
        # 其他参数
        self.num_client = kwargs['num_client']
        self.num_cluster = kwargs['num_cluster']
        self.num_client_per_cluster = self.num_client // self.num_cluster

        self.img_mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
        self.img_std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(os.path.join(self.path, "train"))
            os.makedirs(os.path.join(self.path, "test"))
            trainset, testset, cluster_indices = self.setup_partition(**kwargs)
            self.preprocess(trainset, testset, cluster_indices)
        else:
            with open(os.path.join(self.path, "partition_stat.json"), "r") as f:
                self.partition_stat = json.load(f)

    def setup_partition(self, **kwargs):
        trainset = CIFAR10(self.root, train=True, download=True)
        testset = CIFAR10(self.root, train=False, download=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        trainset.transform = transform
        testset.transform = transform

        cluster_partitioner = CIFAR10Partitioner(
            targets=trainset.targets, num_clients=self.num_cluster,
            balance=kwargs['balance'], partition=kwargs['partition'],
            unbalance_sgm=kwargs['unbalance_sgm'], num_shards=kwargs['num_shards'],
            dir_alpha=kwargs['dir_alpha'], verbose=kwargs['verbose'],
            min_require_size=kwargs['min_require_size'], seed=kwargs['seed']
        )

        return trainset, testset, cluster_partitioner.client_dict


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