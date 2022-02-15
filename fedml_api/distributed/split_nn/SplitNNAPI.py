import torch
from mpi4py import MPI

from fedml_api.distributed.split_nn.client import SplitNN_client
from fedml_api.distributed.split_nn.client_manager import SplitNNClientManager
from fedml_api.distributed.split_nn.server import SplitNN_server
from fedml_api.distributed.split_nn.server_manager import SplitNNServerManager


from fedml_api.distributed.split_nn.facilitator import SplitNN_facilitator # ToDo Changed
from fedml_api.distributed.split_nn.facilitator_manager import SplitNNFacilitatorManager # ToDo Changed

import logging
def SplitNN_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    logging.info("worker_number {}".format(worker_number))
    return comm, process_id, worker_number

def SplitNN_fac_distributed(process_id, worker_number, device, comm, client_model,
                            server_model, facilitator_model, train_data_local, test_data_local, args):
    """
    Instantiates the Guest, Host and Server
    Args:
        process_id: int
        worker_number: int
        device: torch.device
        comm: mpi4py.MPI.Intracomm
        client_model: torch.nn.modules.container.Sequential
        server_model: torch.nn.modules.container.Sequential
        facilitator_model: torch.nn.modules.container.Sequential
        train_data_local: NoneType
        test_data_local: NoneType
        args: argparse.Namespace
    """

    server_rank = 0
    facilitator_rank = 1

    if process_id == server_rank:
        init_server(comm, server_model, process_id, worker_number, device, args)

    elif process_id == facilitator_rank:
        init_facilitator(comm, facilitator_model, process_id, server_rank, worker_number, device, args)

    else:
        init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                    process_id, facilitator_rank, args.epochs, device, args)

def init_facilitator(comm, facilitator_model, process_id, server_rank, worker_number, device, args):
    """
    Instantiates the facilitator
    Args:
        comm: mpi4py.MPI.Intracomm
        facilitator_model: torch.nn.modules.container.Sequential
        process_id: 1 int
        server_rank: 0 int
        worker_number: 4 int
        device: torch.device
        args: argparse.Namespace
    """
    arg_dict = {"comm": comm, "model": facilitator_model, "max_rank": worker_number - 1,
                "rank": process_id,  "server_rank": server_rank, "device": device, "args": args} # ToDo Changed worker number to -2 instead of -1
    logging.info("Step 1: Success facilitator initialized with process ID {}".format(process_id))
    facilitator = SplitNN_facilitator(arg_dict)
    facilitator_manager = SplitNNFacilitatorManager(arg_dict, facilitator)
    facilitator_manager.run()

def init_server(comm, server_model, process_id, worker_number, device, args):
    """
    Instantiates the server
    Args:
        comm: mpi4py.MPI.Intracomm
        server_model: torch.nn.modules.container.Sequential
        process_id: 0 int
        worker_number: 4 int
        device: torch.device
        args: argparse.Namespace
    """

    logging.info("Step 1: Success server initialized with process ID {}".format(process_id))
    arg_dict = {"comm": comm, "model": server_model, "max_rank": worker_number - 1,
                "rank": process_id, "device": device, "args": args} # ToDo Changed worker number to -2 instead of -1
    server = SplitNN_server(arg_dict)
    server_manager = SplitNNServerManager(arg_dict, server)
    server_manager.run()

def init_client(comm, client_model, worker_number, train_data_local, test_data_local,
                process_id, facilitator_rank, epochs, device, args):
    """
    Instantiates the client
    Args:
        comm: mpi4py.MPI.Intracomm
        client_model: torch.nn.modules.container.Sequential
        worker_number: int
        train_data_local: torch.utils.data.dataloader.DataLoader
        test_data_local: torch.utils.data.dataloader.DataLoader
        process_id: 2,3 int
        facilitator_rank: 1 int
        epochs: 4 int
        device: torch.device
        args: argparse.Namespace
    """

    logging.info("Step 1: Success client initialized with process ID {}".format(process_id))
    arg_dict = {"comm": comm, "trainloader": train_data_local, "testloader": test_data_local,
                "model": client_model, "rank": process_id, "facilitator_rank": facilitator_rank,
                "max_rank": worker_number - 1, "epochs": epochs, "device": device, "args": args}
    client = SplitNN_client(arg_dict)
    client_manager = SplitNNClientManager(arg_dict, client)
    client_manager.run()
