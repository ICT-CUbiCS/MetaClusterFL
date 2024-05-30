import sys
import os
from typing import List

from fedavg import FedAvgServer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from client.perfedavg import PerFedAvgClient

class PerFedAvgServer(FedAvgServer):
    def __init__(self) -> None:
        super().__init__()
    
    def setup_clients(self, config, cluster_partitioner) -> List[PerFedAvgClient]:
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
                'hessian_free': config['hessian_free'],
                'per_alpha': config['per_alpha'],
                'per_beta': config['per_beta'],
                'pers_epoch': self.pers_epoch,
            }
            clients.append(PerFedAvgClient(cid, **client_init_args))
        return clients


if __name__ == "__main__":
    server = PerFedAvgServer()
    server.run()