
## 环境准备
安装所需的 python 包： `pip install -r requirements.txt`

## 配置说明
```
{
    "dataset": "CIFAR100",  # 根据data目录下子目录决定
    "model": "lenet",   # 根据./models/dataset目录下子目录决定

    "num_classes": 100, # 取决于dataset数据类别
    "num_client": 100,  # 根据论文中的设置
    "num_cluster": 10,  # 根据论文中的设置

    # 此片段参考 https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html
    "balance": true,
    "partition": "dirichlet",
    "unbalance_sgm": 0.3,
    "num_shards": 50,
    "dir_alpha": 0.25,
    "verbose": true,
    "min_require_size": null,
    "seed": 2023,

    "local_ep": 5,
    "local_bs": 32,
    "lr": 0.01,

    "hessian_free": false,
    "per_beta": 0.001,
    "per_alpha": 0.01,
    "train_client_rate": 0.8,
    "pers_epoch": 1,

    "num_round": 600,
    "eval_interval": 10
}

```

## 代码运行
```shell
cd ./server
python 'algo_name'.py # 按照需要启动不同的算法,例如 python fedavg.py
```


## 目录说明
```
MetaClusterFL
|-- client #不同算法使用到的基本算子client
|   |-- __init__.py
|   |-- fedavg.py
|   |-- perfedavg.py
|-- data #实验数据和数据划分程序的目录
|   |-- CIFAR10 #表示与数据`CIFAR10`相关的文件 
|       |-- partiion #数据划分目录
|           |-- train #训练数据划分目录
|           |-- test #测试数据划分目录
|       |-- raw #原始数据目录
|       |-  partition.py #数据划分程序
|   |-- MNIST #该目录暂时还没有
|       |-- partiion
|           |-- train
|           |-- test
|       |-- raw
|       |-  partition.py
|   |-- ...
|-- models #不同数据集对应的 模型类 目录
|   |-- CIFAR10 #与数据集`CIFAR10`对应的 模型类
|       |- cnn.py
|   |-- MNIST #该目录暂时还没有
|       |- cnn.py
|-- notebook #功能性实验程序目录,使用jupyternotebook类型程序实验
|   |- cifar10_partition.ipynb
|   |- data_operation.ipynb
|   |- grads_difference.ipynb
|   |- inheritance_class_test.ipynb
|   |- model_copy_test.ipynb
|   |- model_test.ipynb
|-- result #训练过程中结果存放目录(刚下载代码时该目录不存在, 代码运行时需要的时候会自动生成)
|   |-- evaluation #模型评估结果
|       |-- server
|   |-- init_args #程序启动参数
|   |-- models #模型
|       |-- client
|       |-- notebook
|       |-- server
|-- server #不同算法实现及启动入口
|   |-- cfl.py
|   |-- fedavg.py
|   |-- flacc.py
|   |-- mcfl.py
|   |-- perfedavg.py
|- __ini__.py
|- .gitattributes
|- .gitignore
|- client_utils.py
|- config.json
|- logging.conf
|- README.md
|- requirements.txt
|- utils.py
```
