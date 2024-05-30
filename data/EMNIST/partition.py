import sys
import os
import json
from torchvision.datasets import EMNIST
from torchvision.transforms import transforms

from fedlab.utils.dataset.partition import CIFAR10Partitioner

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)
from data.utils import dirichlet_partition
from data.CIFAR100.partition import Partition as CIFAR100Partition

class Partition(CIFAR100Partition):
    def __init__(self, **kwargs) -> None:
        # 原始数据存放目录
        self.root = os.path.join(os.path.dirname(__file__), "raw")
        
        # 数据划分存放目录 
        self.path = os.path.join(
            os.path.dirname(__file__), 
            "partition", 
            f"{kwargs['partition']}"
        )
        
        self.num_classes = 62  # 固定值, 取决于数据集本身
        self.trainset_rate = 0.8  # 训练集比例
        self.max_samples_per_class = 50 # 每个客户端最大样本数
        
        # 其他参数
        self.num_client = kwargs['num_client']
        self.num_cluster = kwargs['num_cluster']
        self.num_client_per_cluster = self.num_client // self.num_cluster

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
        trainset = EMNIST(self.root, split='byclass', train=True, download=True)
        testset = EMNIST(self.root, split='byclass', train=False, download=True)
        # 计算数据集的均值和方差
        train_mean = trainset.data.float().mean(axis=(0,1,2)) / 255
        train_std = trainset.data.float().std(axis=(0,1,2)) / 255
        print(f"train_mean: {train_mean}, train_std: {train_std}")
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((train_mean.item(),), (train_std.item(),))
        ])
        trainset.transform = transform
        testset.transform = transform

        if kwargs['partition'] == 'dirichlet':
            clusters_dict = dirichlet_partition(trainset.targets, c_clients=self.num_cluster, alpha=1, n=[4000]*self.num_cluster)
        else:
            cluster_partitioner = CIFAR10Partitioner(
                targets=trainset.targets, num_clients=self.num_cluster,
                balance=kwargs['balance'], partition=kwargs['partition'],
                unbalance_sgm=kwargs['unbalance_sgm'], num_shards=kwargs['num_shards'],
                dir_alpha=kwargs['dir_alpha'], verbose=kwargs['verbose'],
                min_require_size=kwargs['min_require_size'], seed=kwargs['seed']
            )
            clusters_dict = cluster_partitioner.client_dict
        return trainset, testset, clusters_dict

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
    fmnist_partitioner = construct_partition()
    fmnist_partitioner.plot_clusters_labels()
    fmnist_partitioner.plot_random_cluster()
    fmnist_partitioner.plot_clients_labels()