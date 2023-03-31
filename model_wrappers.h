#include kernels.h

// include definitions of model parameters created by make file
#include boilerplate.h
#include model_parameters.h

void launchModel1(cudaStream_t stream, curandState *state, *isng_model_config launch_struct);

void launchModel2(cudaStream_t stream, curandState *state, *isng_model_config launch_struct);

void launchModel3(cudaStream_t stream, curandState *state, *isng_model_config launch_struct);