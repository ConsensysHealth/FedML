# INSTRUCTIONS

# SET UP
1. Create network:
    docker network create fedml-network
2. Create image:
    docker build -t my_image .

# INITIALIZE CLIENT 1
3. Create container:
    docker run --net flwr-network --name client1 -t -d my_image (in a new terminal)
4. Launch client1:
    docker exec -it client1 conda run --no-capture-output -n venv python ./fedml_experiments/distributed/my_fedavg/init_client.py --client_index 1 --client_rank 1

# INITIALIZE CLIENT 2
5. Create container:
    docker run --net flwr-network --name client2 -t -d my_image (in a new terminal)
6. Launch client2:
    docker exec -it client2 conda run --no-capture-output -n venv python ./fedml_experiments/distributed/my_fedavg/init_client.py --client_index 2 --client_rank 2

# INITIALIZE SERVER
7. Create container:
    docker run --net flwr-network --name server -t -d my_image (in a new terminal)
8. Launch server:
    docker exec -it server conda run --no-capture-output -n venv python ./fedml_experiments/distributed/my_fedavg/init_server.py


NOTE: 
1. Most of the parameters are for the moment hard coded.
2. Client and server behaviour controlled by fedml_api/distributed/my_fedavg module.




