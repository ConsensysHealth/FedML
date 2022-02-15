class MyMessage(object):
    """
        message type definition
    """
    # client to client
    MSG_TYPE_S2F_GRADS = 1  # Registered by facilitator send by server
    MSG_TYPE_C2F_SEND_ACTS = 2  # Registered by facilitator send by client
    MSG_TYPE_C2F_VALIDATION_MODE = 3  # Registered by facilitator send by client
    MSG_TYPE_C2F_VALIDATION_OVER = 4  # Registered by facilitator send by client
    MSG_TYPE_C2F_PROTOCOL_FINISHED = 5  # Registered by facilitator send by client
    MSG_TYPE_C2C_SEMAPHORE = 6 # Registered by client and send by client
    MSG_TYPE_F2S_SEND_ACTS = 7  # Registered by server send by facilitator
    MSG_TYPE_F2S_VALIDATION_MODE = 8  # Registered by server send by facilitator
    MSG_TYPE_F2S_VALIDATION_OVER = 9  # Registered by server send by facilitator
    MSG_TYPE_F2S_PROTOCOL_FINISHED = 10  # Registered by server send by facilitator
    MSG_TYPE_F2C_GRADS = 11  # Registered by client send by facilitator

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"