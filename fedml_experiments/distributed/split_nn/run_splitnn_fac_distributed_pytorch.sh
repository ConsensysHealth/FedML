#!/usr/bin/env bash
DATASET=$1

hostname > mpi_host_file

mpirun -np 4 python3 ./main_split_nn_fac.py \
  --client_number 4 \
  --comm_round 2
