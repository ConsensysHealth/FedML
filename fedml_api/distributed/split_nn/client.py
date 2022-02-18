import logging

import torch.optim as optim

class SplitNN_client():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]

        self.trainloader = args["trainloader"]
        self.traindata = self.trainloader[0]
        # ToDo this can be deleted once we have all the data at the server
        self.trainlabel = self.trainloader[1]
        self.train_counter = 0

        self.testloader = args["testloader"]
        self.testdata = self.testloader[0]
        # ToDo this can be deleted once we have all the data at the server
        self.testlabel = self.testloader[1]
        self.test_counter = 0

        self.MAX_RANK = args["max_rank"]
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)
        self.FACILITATOR_RANK = args["facilitator_rank"]
        self.rank = args["rank"]
        self.epoch_count = 0
        self.node_left = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right = 2 if self.rank == self.MAX_RANK else self.rank + 1
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.device = args["device"]

    def forward_pass(self):
        self.optimizer.zero_grad()
        self.acts = self.model(self.data)
        logging.info("rank {} batch_idx {}".format(self.rank, self.batch_idx))
        return self.acts, self.datalabel

    def backward_pass(self, grads):
        self.acts.backward(grads)
        self.optimizer.step()

    def eval_mode(self):
        logging.info("Step 11: Client Entered Eval_mode")
        self.datalabel = self.testlabel[self.test_counter].to(self.device)
        self.data = self.testdata[self.test_counter].to(self.device)
        self.test_counter += 1

        self.model.eval()

    def train_mode(self):
        logging.info("Entered Train_mode")
        self.datalabel = self.trainlabel[self.train_counter].to(self.device)
        logging.info("Entered Train_mode{}".format(self.datalabel))
        self.data = self.traindata[self.train_counter].to(self.device)
        self.train_counter += 1

        self.model.train()
