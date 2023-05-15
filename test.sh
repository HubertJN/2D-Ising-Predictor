#!/bin/bash

gcc calculate_cluster.c functions/* -o bin/calculate_cluster
(cd bin; ./calculate_cluster)
gcc create_cluster_set_v2.c functions/* -o bin/test -g
(cd bin; ./test 1 10000 1)