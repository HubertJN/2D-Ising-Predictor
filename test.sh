#!/bin/bash

gcc calculate_cluster_temp.c functions/* -o bin/calc_temp -g
(cd bin; ./calc_temp)
gcc create_cluster_set_v2.c functions/* -o bin/test -g
(cd bin; ./test 1 10000 1)