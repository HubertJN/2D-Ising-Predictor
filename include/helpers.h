#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "input_reader.h"

// Macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Prototype for gpuAssert
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

void preComputeProbs(ising_model_config *config, float* d_Pacc);

void preComputeNeighbours(ising_model_config *config, int *d_neighbour_list);

#endif // HELPERS_H