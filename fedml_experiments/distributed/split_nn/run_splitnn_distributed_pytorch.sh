#!/usr/bin/env bash
DATASET=$1

hostname > mpi_host_file

mpirun -np 3 -hostfile ./mpi_host_file python3 ./main_split_nn.py \
  --dataset $DATASET \
  --client_number 3 \
  --comm_round 5
