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
parser.add_argument('--client_index', type=int, default=0, metavar='N',
                        help='0 for first client')
parser.add_argument('--client_rank', type=int, default=1, metavar='N',
                        help='1 for first client')


parser.add_argument('--grpc_ipconfig_path', type=str, default='./FedML/fedml_experiments/distributed/my_fedavg/grpc_ipconfig.csv', metavar='N',
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

print('Get data and initialize model... ', end='\r')
transform = transforms.ToTensor()
# Load testset
# data_path = '../../../data'
data_path = './my_data'
testset = MNIST(root=data_path, train=False, download=False, transform=transform)
trainloader = DataLoader(testset, 64)
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

print('Initialize client...', end='\r')
model_trainer = MyModelTrainer(model)

trainer_manager = FedAVGTrainer(client_index=args.client_index,
                            train_local=trainloader,
                            train_data_num=None,
                            device=torch.device('cpu'),
                            args=args,
                            model_trainer=model_trainer)

client = FedAVGClientManager(args = args,
                            trainer = trainer_manager,
                            comm=None,
                            rank=args.client_rank,
                            size=args.client_num_in_total,
                            backend='GRPC')
print('launching client...', end='\r')
client.run()