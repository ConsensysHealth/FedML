from fedml_api.distributed.split_nn.message_define import MyMessage
from fedml_core.distributed.server.server_manager import ServerManager
from fedml_core.distributed.communication.message import Message
import logging

class SplitNNServerManager(ServerManager):

    def __init__(self, arg_dict, trainer, backend="MPI"):
        super().__init__(arg_dict["args"], arg_dict["comm"], arg_dict["rank"],
                         arg_dict["max_rank"] + 1, backend)
        self.trainer = trainer
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        logging.info("Server: Step 4 Saving Messages in a dictionary. From the server this implies that it can handle "
                     "receive msgs from the clinet entailing Send_act, C2S_VALIDATION_MODE, C2S_VALIDATION_OVERC2S_PROTOCOL_FINISHED")
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                                              self.handle_message_validation_over)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)

    def send_grads_to_client(self, receive_id, grads):
        logging.info("Server: Step 8 forwards the gradients to the clients")
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def handle_message_acts(self, msg_params):
        # ToDo for the Facilitator the activations should be just forwarded. So implement a function send_act_to_client
        logging.info("Server: Step 7 send_act forwards the activations and performs backward prop and sends it back to"
                     " client")
        # logging.info("msg_params type{}".format(type(msg_params)))
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.trainer.forward_pass(acts, labels)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            self.send_grads_to_client(self.trainer.active_node, grads)

    def handle_message_validation_mode(self, msg_params):
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        self.trainer.validation_over()

    def handle_message_finish_protocol(self):
        self.finish()
