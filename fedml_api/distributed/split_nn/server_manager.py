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
        self.register_message_receive_handler(MyMessage.MSG_TYPE_F2S_SEND_ACTS,
                                              self.handle_message_acts) # Done
        self.register_message_receive_handler(MyMessage.MSG_TYPE_F2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode) # Done
        self.register_message_receive_handler(MyMessage.MSG_TYPE_F2S_VALIDATION_OVER,
                                              self.handle_message_validation_over) # Done
        self.register_message_receive_handler(MyMessage.MSG_TYPE_F2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)

    def send_grads_to_facilitator(self, grads):
        message = Message(MyMessage.MSG_TYPE_S2F_GRADS, self.get_sender_id(), self.trainer.active_node)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def handle_message_acts(self, msg_params):
        # ToDo change acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.trainer.forward_pass(acts, labels)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            logging.info("Step 6: Server performs back and sends it back to facilitator")
            self.send_grads_to_facilitator(grads)

    def handle_message_validation_mode(self, msg_params):
        logging.info("Step 9a: Received the validation signal from facilitator")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        self.trainer.validation_over()

    def handle_message_finish_protocol(self, msg_params):
        logging.info("Step 17: Received finish signal and is finishing on server side")
        self.finish()
