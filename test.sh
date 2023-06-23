#!/bin/bash

gcc create_cluster_set.c functions/read_input_variables.* functions/comparison.* -o bin/create_cluster_set -Wall -g
#(cd bin; valgrind ./create_cluster_set 1000)
(cd bin; ./create_cluster_set 1000)