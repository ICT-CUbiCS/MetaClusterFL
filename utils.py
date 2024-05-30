import os
import json
import importlib

PROJECT_DIR = os.path.dirname(__file__)

def read_options():
    
    with open(os.path.join(PROJECT_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    
    partition_path = 'data.{}.partition'.format(config['dataset'])
    partition = importlib.import_module(partition_path)
    partitioner = getattr(partition, 'Partition')
    partitioner_args = {
        'num_client': config['num_client'],
        'num_cluster': config['num_cluster'],
        
        'balance': config['balance'],
        'partition': config['partition'],
        'unbalance_sgm': config['unbalance_sgm'],
        'num_shards': config['num_shards'],
        'dir_alpha': config['dir_alpha'],
        'verbose': config['verbose'],
        'min_require_size': config['min_require_size'],
        'seed': config['seed'],
        
        # 'accelerate': config['accelerate']
    }
    cluster_partitioner = partitioner(**partitioner_args)
    
    model_path = 'models.{}.{}'.format(config['dataset'], config['model'])
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')
    
    return config, cluster_partitioner, model