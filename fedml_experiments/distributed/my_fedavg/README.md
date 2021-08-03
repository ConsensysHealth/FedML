# INSTRUCTIONS

# SET UP
1. Create docker container:
    docker build -t my_image .
2. Run image:
    docker run -t -d my_image
3. Retrieve CONTAINER ID:
    docker ps

# INITIALIZE CLIENTS
1. Open new terminal and run:
    docker exec -it [CONTAINER_ID] conda run --no-capture-output -n venv python ./fedml_experiments/distributed/my_fedavg/init_client.py --client_index 0 --client_rank 1
2. Open new terminal and run:
    docker exec -it [CONTAINER_ID] conda run --no-capture-output -n venv python ./fedml_experiments/distributed/my_fedavg/init_client.py --client_index 1 --client_rank 2

# INITIALIZE SERVER
1. Open new terminal and run:
    docker exec -it [CONTAINER_ID] conda run --no-capture-output -n venv python ./fedml_experiments/distributed/my_fedavg/init_server.py

NOTE: 
1. Most of the parameters are for the moment hard coded.
2. Client and server behaviour controlled by fedml_api/distributed/my_fedavg module.




