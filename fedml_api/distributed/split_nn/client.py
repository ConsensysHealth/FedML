import logging
import torch
import torch.optim as optim

class SplitNN_client():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]

        self.traindata = args["trainloader"]
        self.train_counter = 0

        self.testdata = args["testloader"]
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

    def forward_pass(self):# -> tuple[torch.Tensor, list]:
        if self.phase == "train":
            self.data = self.traindata[self.train_counter].to(self.device)
            self.counter = self.train_counter
            self.train_counter += 1
            logging.info("self.train_counter {} and self.rank {} ".format(self.train_counter,self.rank))
        else:
            self.data = self.testdata[self.test_counter].to(self.device)
            self.counter = self.test_counter
            self.test_counter += 1
            logging.info("self.test_counter {} and self.rank {} ".format(self.test_counter,self.rank))
        self.optimizer.zero_grad()
        self.acts = self.model(self.data)
        logging.info("rank {} batch_idx {}".format(self.rank, self.batch_idx))
        return self.acts, (self.rank,self.counter,self.phase)

    def backward_pass(self, grads: torch.Tensor):
        self.acts.backward(grads)
        self.optimizer.step()

    def eval_mode(self):
        logging.info("Step 11: Client Entered Eval_mode with test_counter {} of rank {}".format(self.test_counter,
                                                                                                self.rank))
        self.phase = "evaluation"
        self.model.eval()

    def train_mode(self):
        logging.info("Entered Train_mode")
        self.phase = "train"
        self.model.train()
