from mpi4py import MPI

from fedml_api.distributed.split_nn.client import SplitNN_client
from fedml_api.distributed.split_nn.client_manager import SplitNNClientManager
from fedml_api.distributed.split_nn.server import SplitNN_server
from fedml_api.distributed.split_nn.server_manager import SplitNNServerManager

from fedml_api.distributed.split_nn.facilitator import SplitNN_facilitator
from fedml_api.distributed.split_nn.facilitator_manager import SplitNNFacilitatorManager
import logging

def SplitNN_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number

def SplitNN_distributed(process_id, worker_number, device, comm, client_model,
                        server_model, train_data_num, train_data_global, test_data_global,
                        local_data_num, train_data_local, test_data_local, args):
# ToDo def SplitNN_distributed(process_id, worker_number, device, comm, client_model,
#                         server_model, facilitator_model,
#                         train_data_num, train_data_global, test_data_global,
#                         local_data_num, train_data_local, test_data_local, args):
    server_rank = 0
    facilitator_rank = 1
    if process_id == server_rank:
        init_server(comm, server_model, process_id,  worker_number, device, args)
    # ToDo elif process_id == facilitator_rank:
    #     init_facilitator(comm, facilitator_model, process_id, server_rank, worker_number, device, args)
    else:
        # ToDo check if this should be facilitator rank instead
        init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                    process_id, server_rank, args.epochs, device, args)

def init_server(comm, server_model, process_id, worker_number, device, args):
    logging.info("Server: Step 1 Instantiating SplitNN_distributed")

    # ToDo understand max_rank if it potentially need to be adjusted to -2
    arg_dict = {"comm": comm, "model": server_model, "max_rank": worker_number - 1,
                "rank": process_id, "device": device, "args": args}
    logging.info("Server: Step 2 Instantiates SplitNN_distributed")
    server = SplitNN_server(arg_dict)
    logging.info("Server: Step 3 Instantiates SplitNN ServerManager who inherits from Server Manager")
    server_manager = SplitNNServerManager(arg_dict, server)
    server_manager.run()

# ToDo def init_facilitator(comm, facilitator_model, process_id, server_rank, worker_number, device, args):
#     logging.info("Server: Step 1 Instantiating SplitNN_distributed")
#     arg_dict = {"comm": comm, "model": facilitator_model, "max_rank": worker_number - 1,
#                 "rank": process_id,  "server_rank": server_rank, "device": device, "args": args}
#     facilitator = SplitNN_facilitator(arg_dict)
#     facilitator_manager = SplitNNFacilitatorManager(arg_dict, facilitator)
#     facilitator_manager.run()

def init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                process_id, server_rank, epochs, device, args):
    # ToDo check if server_rank should be facilitator rank instead
    # ToDo understand max_rank if it potentially need to be adjusted to -2
    logging.info("Client {}: Step 1 Instantiating SplitNN_distributed".format(process_id))

    arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
                "model": client_model, "rank": process_id, "server_rank": server_rank,
                "max_rank": worker_number - 1, "epochs": epochs, "device": device, "args": args}
    logging.info("Client {}: Step 2 Instantiates SplitNN_distributed".format(process_id))
    client = SplitNN_client(arg_dict)
    logging.info("Client {}: Step 3 Instantiates SplitNN ClientManager who inherits from Client Manager".format(process_id))
    client_manager = SplitNNClientManager(arg_dict, client)
    client_manager.run()
