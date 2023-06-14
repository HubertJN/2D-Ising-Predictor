#ifndef MODEL_WRAPPERS_H
#define MODEL_WRAPPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.h"
#include "input_reader.h"
#include "helpers.h"

void init_model(ising_model_config* launch_struct);

void launch_mc_sweep(cudaStream_t stream, curandState *state, ising_model_config* launch_struct, int *device_array);



#endif // MODEL_WRAPPERS_H
