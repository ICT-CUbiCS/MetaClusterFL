import sys
import os
import logging
import threading
import random
from typing import List
import numpy as np
import pandas as pd
import torch

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from utils import read_options
from client.fedavg import FedAvgClient

class FedAvgServer():
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)
        config, cluster_partitioner, model_constructor = read_options()
        
        self.global_model: torch.nn.Module = model_constructor()
        # save initial global model
        initial_global_model_dir = os.path.join(PROJECT_DIR, 'result', 'models', 'init', config['dataset'])
        if not os.path.exists(initial_global_model_dir):
            os.makedirs(initial_global_model_dir)
        initial_global_model_path = os.path.join(initial_global_model_dir, "{}_init_global_model.pth".format(config['model']))
        if not os.path.exists(initial_global_model_path):
            torch.save(self.global_model.state_dict(), initial_global_model_path)
        else:
            self.global_model.load_state_dict(torch.load(initial_global_model_path))
        
        self.num_round = config['num_round']
        self.seed = config['seed']
        self.eval_interval = config['eval_interval']
        self.pers_epoch = config['pers_epoch']

        self.clients: List[FedAvgClient] = self.setup_clients(config, cluster_partitioner)
        self.clients_sample_stream: List[List[FedAvgClient]] = self.setup_sample_stream()

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

    def setup_sample_stream(self):
        sample_stream = []
        random.seed(self.seed)
        for _ in range(self.num_round):
            sample_stream.append(random.sample(self.clients, max(1, int(len(self.clients) * 0.2))))
        return sample_stream

    def aggregate(self, selected_clients: List[FedAvgClient]): 
        selected_clients_param_cache = [SerializationTool.serialize_model(client.local_model) for client in selected_clients]
        avg_param = Aggregators.fedavg_aggregate(selected_clients_param_cache)
        SerializationTool.deserialize_model(self.global_model, avg_param)
    
    def save_model(self):
        model_dir = os.path.join(PROJECT_DIR, 'result', 'models', 'server')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'global_model_in_{}.pth'.format(self.__class__.__name__))
        torch.save(self.global_model.state_dict(), model_path)
    
    def save_eval_result(self, result_df: pd.DataFrame):
        result_dir = os.path.join(PROJECT_DIR, 'result', 'evaluation', 'server')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'eval_result_in_{}.csv'.format(self.__class__.__name__))
        result_df.to_csv(result_path)

    def init_eval_result(self):
        columns = ['round', 'loss', 'top1', 'top5']
        df = pd.DataFrame(columns=columns)
        df = df.set_index(columns[0])
        return df

    def evaluate(self):
        results = [client.evaluate(self.global_model) for client in self.clients]
        loss_list, top1_list, top5_list = zip(*results)
        loss, top1, top5 = np.mean(loss_list), np.mean(top1_list), np.mean(top5_list)
        return loss, top1, top5

    def run(self):
        eval_result = self.init_eval_result()
        for current_round in range(self.num_round):
            selected_clients = self.clients_sample_stream[current_round]
            self.logger.info(f"Round {current_round + 1} selected clients: {[client.cid for client in selected_clients]}")
            for client in selected_clients:
                client.train(self.global_model)
            self.aggregate(selected_clients)

            if (current_round + 1) % self.eval_interval == 0:
                # save model
                thread = threading.Thread(target=self.save_model)
                thread.start()

                # evaluate on all clients
                self.logger.info(f"Round {current_round + 1} evaluation")
                loss, top1, top5 = self.evaluate()
                eval_result.loc[current_round + 1] = [loss, top1, top5]
                self.logger.info(f"Round {current_round + 1} evaluation result: loss {loss}, top1 {top1}, top5 {top5}")
                thread = threading.Thread(target=self.save_eval_result, args=(eval_result,))
                thread.start()

if __name__ == "__main__":
    server = FedAvgServer()
    server.run()