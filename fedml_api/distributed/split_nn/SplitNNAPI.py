import torch
from mpi4py import MPI

from fedml_api.distributed.split_nn.client import SplitNN_client
from fedml_api.distributed.split_nn.client_manager import SplitNNClientManager
from fedml_api.distributed.split_nn.server import SplitNN_server
from fedml_api.distributed.split_nn.server_manager import SplitNNServerManager
from torch.nn import Sequential

from fedml_api.distributed.split_nn.facilitator import SplitNN_facilitator # ToDo Changed
from fedml_api.distributed.split_nn.facilitator_manager import SplitNNFacilitatorManager # ToDo Changed

import logging

def SplitNN_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    logging.info("worker_number {}".format(worker_number))
    return comm, process_id, worker_number

def SplitNN_fac_distributed(process_id: int, worker_number: int, device, comm, client_model,
                            server_model: Sequential, facilitator_model: Sequential, train_data_batch,
                            train_label_batch, test_data_batch, test_label_batch, args):
    """
    Instantiates the Guest, Host and Facilitator

    Args:
        process_id: IDs that are assigned to each participating client in the federated learning env. 0 is assigned to
        Guest by default, 1 to the Facilitator and the remaining to the participating hosts.
        worker_number: Overall number of participants in the FL env
    """

    server_rank = 0
    facilitator_rank = 1

    if process_id == server_rank:
        train_label_batch = None
        test_label_batch = None
        init_server(comm, server_model, process_id, worker_number, train_label_batch, test_label_batch, device, args)
    elif process_id == facilitator_rank:
        init_facilitator(comm, facilitator_model, process_id, server_rank, worker_number, device, args)
    else:
        # ToDo Delete this label laaaaaaaaater :)
        init_client(comm, client_model, worker_number, (train_data_batch, train_label_batch),
                    (test_data_batch,train_label_batch),process_id, facilitator_rank, args.epochs, device, args)

def init_facilitator(comm, facilitator_model: Sequential, process_id: int, server_rank: int, worker_number: int,
                     device, args):
    arg_dict = {"comm": comm, "model": facilitator_model, "max_rank": worker_number - 1,
                "rank": process_id,  "server_rank": server_rank, "device": device, "args": args}
    logging.info("Step 1: Facilitator initialized with process ID {}".format(process_id))
    facilitator = SplitNN_facilitator(arg_dict)
    facilitator_manager = SplitNNFacilitatorManager(arg_dict, facilitator)
    facilitator_manager.run()

def init_server(comm, server_model: Sequential, process_id: int, worker_number: int, train_label, test_label, device, args):
    # ToDo update train_label, test_label type
    logging.info("Step 1: Server initialized with process ID {}".format(process_id))
    arg_dict = {"comm": comm, "trainlabel": train_label, "testlabel": test_label,"model": server_model, "max_rank": worker_number - 1,
                "rank": process_id, "device": device, "args": args}
    server = SplitNN_server(arg_dict)
    server_manager = SplitNNServerManager(arg_dict, server)
    server_manager.run()

def init_client(comm, client_model: Sequential, worker_number:int, train_data_local, test_data_local,
                process_id: int, facilitator_rank: int, epochs: int, device, args):
    # ToDo update train_label, test_label type
    logging.info("Step 1: Client initialized with process ID {}".format(process_id))
    arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
                "model": client_model, "rank": process_id, "facilitator_rank": facilitator_rank,
                "max_rank": worker_number - 1, "epochs": epochs, "device": device, "args": args}
    client = SplitNN_client(arg_dict)
    client_manager = SplitNNClientManager(arg_dict, client)
    client_manager.run()
