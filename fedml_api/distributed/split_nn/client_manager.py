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
        self.arg_dict = arg_dict

    def run(self):
        # ToDo maybe this trainer.ramk == 2 if we add the facilitator
        if self.trainer.rank == 1:
            logging.info("Starting protocol from rank {} process".format(self.trainer.rank))
            self.run_forward_pass()
        super().run()

    def register_message_receive_handlers(self):
        logging.info("Client {}: Step 4 Saving Messages in a dictionary. From the client this implies that it can handle "
                     "C2C amd S2C grads".format(self.arg_dict["rank"]))
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)

    def handle_message_semaphore(self, msg_params):
        # no point in checking the semaphore message
        logging.info("Starting training at node {}".format(self.trainer.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.round_idx += 1
        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
                self.send_finish_to_server(self.trainer.SERVER_RANK)
        else:
            logging.info("sending semaphore from {} to {}".format(self.trainer.rank,
                                                                  self.trainer.node_right))
            self.send_semaphore_to_client(self.trainer.node_right)

        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE:
            self.finish()

    def handle_message_gradients(self, msg_params):
        logging.info("Client {}: Step 9 receives the gradients".format(self.trainer.rank))
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            logging.info("Epoch over at node {}".format(self.rank))
            self.round_idx += 1
            self.run_eval()
        else:
            self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        # ToDo adjust that clients send to Facilitator
        logging.info("Client {}: Step 5 sends the activations and labels to receive_id {}".format(self.trainer.rank,receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.get_sender_id(), receive_id)
        # ToDo adjust that clients send only activations and not labels
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.get_sender_id(), receive_id) 
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.get_sender_id(), receive_id)
        self.send_message(message)
