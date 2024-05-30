## CIFAR10数据集
CIFAR-10（Canadian Institute for Advanced Research 10）是一个常用的计算机视觉数据集，用于图像分类任务。它由10个不同类别的彩色图像组成，每个类别有6000张图像，总共有60000张图像。

数据格式：

CIFAR-10数据集的数据格式通常采用常见的图像数据表示方式，图像数组和标签数
1. 图像数据格式：每个图像都以图像数组的形式表示。图像数组是一个三维数组，其形状为[height, width, channels]。对于CIFAR-10数据集的图像，其高度（height）和宽度（width）均为32个像素，而通道数（channels）为3，表示红色、绿色和蓝色三个通道。因此，每个图像的图像数组的形状为[32, 32, 3]。
2. 标签数据格式：每个图像都有一个与之对应的标签，用于表示图像所属的类别。标签通常以整数形式表示，范围从0到9，对应于CIFAR-10数据集的10个类别。

特点：
1. 图像类别：CIFAR-10数据集包含了10个不同的类别，分别是飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、青蛙（frog）、马（horse）、船（ship）和卡车（truck）。
2. 图像内容：每个类别的图像都是32x32像素大小的彩色图像，即RGB格式。这些图像包含了各自类别的实际物体或场景的视觉表示。
3. 训练集和测试集：CIFAR-10数据集被划分为训练集和测试集。训练集包含了50000张图像，用于模型的训练和参数优化。测试集包含了10000张图像，用于评估模型在未见过的数据上的性能。
4. 数据标签：每个图像都有一个与之关联的标签，表示它所属的类别。标签是从0到9的整数，分别对应10个类别。
5. 数据规模和挑战性：CIFAR-10数据集相对较小，但是由于图像分辨率较低且类别之间视觉特征差异较小，因此仍然是一个具有挑战性的数据集。它被广泛用于测试和比较各种图像分类算法的性能。

## partition.py
通过FedLab框架的CIFAR10Partitione类分区并统计CIFAR10数据集，支持多种划分方法(通过参数balance和partition控制)，支持多进程和单线程。

核心代码:
```python
cluster_partitioner = CIFAR10Partitioner(
                targets=trainset.targets, num_clients=self.num_cluster,
                balance=kwargs['balance'], partition=kwargs['partition'],
                unbalance_sgm=kwargs['unbalance_sgm'], num_shards=kwargs['num_shards'],
                dir_alpha=kwargs['dir_alpha'], verbose=kwargs['verbose'], 
                min_require_size=kwargs['min_require_size'] ,seed=kwargs['seed']
            )
``` 

## CIFAR10数据划分
FedLab为CIFAR10提供了6种预定义的数据划分方案。根据以下参数来划分CIFAR10：
- targets 是数据集对应的标签

- num_clients 指定划分方案中的client数量

- balance 指不同client的数据样本数量全部相同的联邦学习场景

- partition 指定划分方法的名字

- unbalance_sgm 是用于非均衡划分的参数

- num_shards 是基于shards进行non-IID划分需要的参数

- dir_alpha 是划分中用到的Dirichlet分布需要的参数

- verbose 指定是否打印中间执行信息

- seed 用于设定随机种子
 
划分方法：
1. partition="dirichlet"：
   partition="dirichlet"表示采用狄利克雷过程（Dirichlet Process）进行划分。具体而言，狄利克雷过程是一种随机过程，它将不确定性引入到聚类中，并通过引入随机性来模拟群集数据的动态增长过程。使用狄利克雷过程对样本进行划分时，每个客户端都会被分配一个多项式分布（Multinomial distribution），该分布定义了一个概率向量，表示各个类别在该客户端上出现的概率，概率值的总和为1。客户端的数据被从CIFAR10训练集中按照这个多项式分布进行抽样，从而使得每个客户端中各个类别的样本数量服从一个概率分布。当partition="dirichlet"时，还需要设置参数dir_alpha，该参数控制所使用的狄利克雷分布的参数α，值越小则客户端之间的差异越大。具体来说：
   - partition="dirichlet"：使用狄利克雷过程对数据进行划分
   - dir_alpha：狄利克雷分布的参数α，值越小则客户端之间的差异越大
   - balance：None-不预先指定每个客户端样本数量，True-均匀分配样本，False-随机分配
2. partition="shards"：
partition="shards"表示采用分片（Shards）的方法进行划分。具体而言，该方法将CIFAR10训练集中的数据随机分成多个数据块，每个数据块包含相同数量的训练样本。可以通过传递参数num_shards来指定数据块的数量。然后，通过将每个数据块分配给不同的客户端来将训练数据划分为不同的客户端。
3. partition="iid"：
partition="iid"表示采用独立同分布（Independent and Identically Distributed，IID）的方法将训练数据划分到不同的客户端中。具体而言，该方法随机地将CIFAR10训练集中的样本分配给不同的客户端，每个客户端获得的样本是相互独立的、具有相同的分布。在这种情况下，客户端之间的数据是相互独立的，并且没有相关性。当partition="iid"时，还需要设置参数balance，该参数用于控制是否对训练数据进行负载均衡。
    - balance=True，会将CIFAR10数据集中的样本均匀分配给各个客户端
    - balance=False，采用随机分配的方式将部分数据集分配给不同的客户端，有些客户端可能获得的数据集规模较小