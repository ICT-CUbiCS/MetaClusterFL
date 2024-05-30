import sys
import os
from mcfl import MetaClusterFLServer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from client.fedavg import FedAvgClient

class NoMetaClusterFLServer(MetaClusterFLServer):
    """
    NoMetaClusterFLServer is a FL server that does not use meta-clustering.
    """
    def __init__(self):
        super().__init__()
    
    def setup_clients(self, config, cluster_partitioner):
        num_clients = config['num_client']
        batch_size = config['local_bs']
        clients = []
        for cid in range(num_clients):
            client_init_args = {
                'logger': self.logger,
                'model_template': self.global_model,
                'trainloader': cluster_partitioner.get_dataloader(cid, batch_size),
                'testloader': cluster_partitioner.get_dataloader(cid, batch_size, type='test', cluster=True),
                'local_epoch': config['local_ep'],
                'lr': config['lr'],
            }
            clients.append(FedAvgClient(cid, **client_init_args))
        return clients


if __name__ == '__main__':
    server = NoMetaClusterFLServer()
    server.run()