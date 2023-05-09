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

void testModel1(cudaStream_t stream, curandState *state, ising_model_config launch_struct);
void testModel2(cudaStream_t stream, curandState *state, ising_model_config launch_struct);

#endif // MODEL_WRAPPERS_H
