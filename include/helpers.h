#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "input_reader.h"

void preComputeProbs(ising_model_config *config, float* d_Pacc);

void preComputeNeighbours(ising_model_config *config, int *d_neighbour_list);

#endif // HELPERS_H