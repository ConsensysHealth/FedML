import torch

from fedml_api.distributed.split_nn.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

import logging

class SplitNNClientManager(ClientManager):

    def __init__(self, arg_dict, trainer, backend="MPI"):
        super().__init__(arg_dict["args"], arg_dict["comm"], arg_dict["rank"],
                         arg_dict["max_rank"] + 1, backend)
        self.trainer = trainer
        self.trainer.train_mode()
        self.round_idx = 0

    def run(self):
        if self.trainer.rank == 2:
            logging.info("Step 2: Starting Training from client rank 2 in training mode")
            self.run_forward_pass()
        super().run()

    def register_message_receive_handlers(self):
        # This is send from one client to another one in send_semaphore_to_client
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore) # Done
        # This is send facilitator send_grads_to_client
        self.register_message_receive_handler(MyMessage.MSG_TYPE_F2C_GRADS,
                                              self.handle_message_gradients) # Done

    def run_forward_pass(self):
        acts, rank = self.trainer.forward_pass()
        logging.info("Step 3/12: Client {} run_forward_pass on batch {}".format(self.trainer.rank, self.trainer.batch_idx))
        self.send_activations_to_facilitator(acts, rank, self.trainer.FACILITATOR_RANK)
        self.trainer.batch_idx += 1


    def send_activations_to_facilitator(self, acts: torch.Tensor, rank: int, receive_id: int):
        logging.info("Step 4: Client  rank {} sends activation and label to facilitator {}".format(self.get_sender_id(), receive_id))
        message = Message(MyMessage.MSG_TYPE_C2F_SEND_ACTS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, rank))
        self.send_message(message)

    def handle_message_semaphore(self, msg_params):
        logging.info("Starting training at node {}".format(self.trainer.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

    def run_eval(self):
        self.send_validation_signal_to_facilitator(self.trainer.FACILITATOR_RANK)
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testdata)):
            self.run_forward_pass()
        self.trainer.test_counter = 0
        self.send_validation_over_to_facilitator(self.trainer.FACILITATOR_RANK)
        self.round_idx += 1
        if self.round_idx == (self.trainer.MAX_EPOCH_PER_NODE) and self.trainer.rank == self.trainer.MAX_RANK:
                logging.info("Step 15: Start finish signal"
                             .format(self.round_idx, self.trainer.MAX_EPOCH_PER_NODE, self.trainer.rank,
                                     self.trainer.MAX_RANK))
                self.send_finish_to_facilitator(self.trainer.FACILITATOR_RANK)
        else:
            logging.info("Step 14: Client starts training at the other client and 2-13 repeat. round {} "
                         "self.trainer.MAX_EPOCH_PER_NODE {} self.trainer.rank {} self.trainer.MAX_RANK {}"
                         .format(self.round_idx, self.trainer.MAX_EPOCH_PER_NODE, self.trainer.rank,
                                 self.trainer.MAX_RANK))
            self.send_semaphore_to_client(self.trainer.node_right)


        # if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE:
        #     self.finish()

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        if self.trainer.batch_idx == len(self.trainer.traindata):

            logging.info("Step 7 Alternative: Client received grads for evaluation Epoch over at node {}".format(self.rank))
            #self.round_idx += 1 ToDo Change 2
            self.run_eval()
            self.trainer.train_counter = 0 # ToDo Change 2
            self.trainer.batch_idx = 0
        else:
            logging.info("Step 7: Client {} received grads performed back and is performing forward of batch_idx {} "
                         "/ len trainloader {}".format(self.rank,self.trainer.batch_idx,len(self.trainer.traindata)))
            self.run_forward_pass()

    def send_semaphore_to_client(self, receive_id: int):
        logging.info("sending semaphore from {} to {}".format(self.trainer.rank, self.trainer.node_right))
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_validation_signal_to_facilitator(self, receive_id: int):
        logging.info("Step 8 Alternative: Send val signal to facilitator")
        message = Message(MyMessage.MSG_TYPE_C2F_VALIDATION_MODE, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_validation_over_to_facilitator(self, receive_id: int):
        message = Message(MyMessage.MSG_TYPE_C2F_VALIDATION_OVER, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_finish_to_facilitator(self, receive_id: int):
        message = Message(MyMessage.MSG_TYPE_C2F_PROTOCOL_FINISHED, self.get_sender_id(), receive_id)
        self.send_message(message)
