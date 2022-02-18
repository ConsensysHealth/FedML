import logging

import torch.nn as nn
import torch.optim as optim
import torch

class SplitNN_server():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]
        self.MAX_RANK = args["max_rank"]


        self.trainlabel = args["trainlabel"]
        self.train_counter = 0

        self.testlabel = args["testlabel"]
        self.test_counter = 0
        self.rank = 2

        self.init_params()


    def init_params(self):
        self.epoch = 0
        self.log_step = 50
        self.active_node = 1
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        logging.info("Step 10: Evaluates model and resets parameters to enter train phase again self.rank {} "
                     "self.test_counter {}".format(self.rank,self.test_counter))
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts: torch.Tensor, id_info:list):
        rank = id_info[0]
        counter = id_info[1]
        phase = id_info[2]
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        logits = self.model(self.acts)
        _, predictions = logits.max(1)
        logging.info(
            "phase {} rank {} train_counter {} ".format(phase, rank, counter))
        if phase == "train":
            labels = self.trainlabel[rank][counter]
        else:
            labels = self.testlabel[rank][counter]

        self.loss = self.criterion(logits, labels)
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()

        if self.step % self.log_step == 0 and self.phase == "train":
            logging.info("Step 5: Server performs forward on activation")
            acc = self.correct / self.total
            logging.info("phase={} acc={} loss={} epoch={} and step={}"
                         .format("train", acc, self.loss.item(), self.epoch, self.step))
        if self.phase == "validation":
            self.val_loss += self.loss.item()
        self.step += 1

    def backward_pass(self) -> torch.Tensor:
        self.loss.backward()
        self.optimizer.step()
        return self.acts.grad

    def validation_over(self):
        # not precise estimation of validation loss
        self.val_loss /= self.step
        acc = self.correct / self.total
        logging.info("phase={} acc={} loss={} epoch={} and step={}"
                     .format(self.phase, acc, self.val_loss, self.epoch, self.step))
        self.epoch += 1
        self.train_mode()
        logging.info("Step 13 Server received validation over and sets it back to train mode")
