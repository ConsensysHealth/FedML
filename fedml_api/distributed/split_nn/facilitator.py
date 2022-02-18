import logging

import torch
import torch.nn as nn
import torch.optim as optim

class SplitNN_facilitator():

    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]
        self.MAX_RANK = args["max_rank"]
        self.init_params()
        self.SERVER_RANK = args["server_rank"]

    def init_params(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)
        self.active_node = 2
        self.epoch = 0
        self.train_mode()

    def reset_local_params(self):
        self.batch_idx = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts: torch.Tensor, rank_info: list): #-> tuple[torch.Tensor,list]:
        self.optimizer.zero_grad()
        self.input_layer = acts
        self.input_layer.retain_grad()
        self.acts = self.model(acts)
        self.acts.retain_grad()
        return self.acts, rank_info

    def backward_pass(self, grads: torch.Tensor) -> torch.Tensor:
        self.acts.backward(grads)
        self.optimizer.step()
        return self.input_layer.grad

    def validation_over(self):
        # not precise estimation of validation loss
        self.active_node = ((self.active_node - 1) % (self.MAX_RANK - 1)) + 1
        self.active_node += 1
        self.train_mode()
        logging.info("Step 12 validating current active client{}".format(self.active_node))