#ifndef MODEL_WRAPPERS_H
#define MODEL_WRAPPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.h"
#include "input_reader.h"

void launchModel1(cudaStream_t stream, curandState *state, ising_model_config launch_struct);
void launchModel2(cudaStream_t stream, curandState *state, ising_model_config launch_struct);
void launchModel3(cudaStream_t stream, curandState *state, ising_model_config launch_struct);

int testModel1(cudaStream_t stream, curandState *state, ising_model_config launch_struct, int *device_array);
int testModel2(cudaStream_t stream, curandState *state, ising_model_config launch_struct, int *device_array);

#endif // MODEL_WRAPPERS_H
