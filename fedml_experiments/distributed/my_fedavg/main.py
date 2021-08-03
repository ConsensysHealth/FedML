import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


try:
    from fedml_api.distributed.my_fedavg.MyModelTrainer import MyModelTrainer
    from fedml_api.distributed.my_fedavg.FedAVGAggregator import FedAVGAggregator
    from fedml_api.distributed.my_fedavg.FedAvgServerManager import FedAVGServerManager
    from fedml_api.distributed.my_fedavg.FedAvgClientManager import FedAVGClientManager
    from fedml_api.distributed.my_fedavg.FedAVGTrainer import FedAVGTrainer
except:
    from FedML.fedml_api.distributed.my_fedavg.MyModelTrainer import MyModelTrainer
    from FedML.fedml_api.distributed.my_fedavg.FedAVGAggregator import FedAVGAggregator
    from FedML.fedml_api.distributed.my_fedavg.FedAvgServerManager import FedAVGServerManager
    from FedML.fedml_api.distributed.my_fedavg.FedAvgClientManager import FedAVGClientManager
    from FedML.fedml_api.distributed.my_fedavg.FedAVGTrainer import FedAVGTrainer

# args
parser = argparse.ArgumentParser(description='server communication arguments')
parser.add_argument('--grpc_ipconfig_path', type=str, default='./fedml_experiments/distributed/my_fedavg/grpc_ipconfig.csv', metavar='N',
                        help='neural network used in training')
parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')
parser.add_argument('--is_mobile', type=str, default=1, metavar='N',
                        help='are the clients running on mobile devices')
parser.add_argument('--client_num_in_total', type=str, default=2, metavar='N',
                        help='number of clients in total')
parser.add_argument('--client_num_per_round', type=str, default=2, metavar='N',
                        help='number of clients per round')
parser.add_argument('--frequency_of_the_test', type=str, default=1, metavar='N',
                        help='')
parser.add_argument('--comm_round', type=str, default=2, metavar='N',
                        help='number of FL rounds')
parser.add_argument('--ci', type=str, default=0, metavar='N',
                        help='no idea')
parser.add_argument('--client_optimizer', type=str, default='sgd', metavar='N',
                        help='')
parser.add_argument('--lr', type=str, default=0.001, metavar='N',
                        help='learning rate')
parser.add_argument('--epochs', type=str, default=1, metavar='N',
                        help='')
args = parser.parse_args()

print('Get data and initialize model... ')
transform = transforms.ToTensor()
# Load testset
# data_path = '../../../data'
print('flag')
data_path = './my_data'
testset = MNIST(root=data_path, train=False, download=False, transform=transform)
trainloader1 = DataLoader(testset, 64)
trainloader2 = DataLoader(testset, 64)
testloader = DataLoader(testset, 32)

# Create model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
model1 = Net()
model2 = Net()

print('Initialize clients...')
model_trainer1 = MyModelTrainer(model1)
model_trainer2 = MyModelTrainer(model2)

trainer_manager1 = FedAVGTrainer(client_index=0,
                            train_local=trainloader1,
                            train_data_num=None,
                            device=torch.device('cpu'),
                            args=args,
                            model_trainer=model_trainer1)
trainer_manager2 = FedAVGTrainer(client_index=1,
                            train_local=trainloader2,
                            train_data_num=None,
                            device=torch.device('cpu'),
                            args=args,
                            model_trainer=model_trainer2)

client1 = FedAVGClientManager(args = args,
                            trainer = trainer_manager1,
                            comm=None,
                            rank=1,
                            size=args.client_num_in_total,
                            backend='GRPC')
client2 = FedAVGClientManager(args = args,
                            trainer = trainer_manager2,
                            comm=None,
                            rank=2,
                            size=args.client_num_in_total,
                            backend='GRPC')


print('Initialize server...')
evaluator = MyModelTrainer(model)

aggregator = FedAVGAggregator(train_global = None,
                        test_global = testloader,
                        all_train_data_num = None,
                        train_data_local_dict = None,
                        test_data_local_dict = data_path,
                        train_data_local_num_dict = None,
                        worker_num = 2,
                        device = torch.device('cpu'),
                        args = args,
                        model_trainer = evaluator)

server_manager = FedAVGServerManager(args = args,
                                aggregator = aggregator,
                                comm=None,
                                rank=0,
                                size=3,
                                backend="GRPC",
                                is_preprocessed=False,
                                preprocessed_client_lists=None)

print('launching client1')
client1.run()
print('launching client2')
client2.run()
print('launching server')
server_manager.send_init_msg()
server_manager.run()