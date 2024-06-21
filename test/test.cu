#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#include "../include/kernels.h"
#include "../include/input_reader.h"
#include "../include/model_wrappers.h"

int main( int argc, char *argv[]) {

    // Single Kernel test.

    printf("Single Kernel Test\n");

    ising_model_config config;

    //Misc Vars
    int test_threads = 5;

    config.model_id = 1;
    config.num_concurrent = test_threads;

    // Initialize RNG
    curandState_t *d_rng_state;
    // Allocate each thread memory for a random state 
    cudaMalloc((void **)&d_rng_state, test_threads * sizeof(curandState)); 

    init_rng<<<1, test_threads>>>(d_rng_state, time(NULL), test_threads);

    // run the test kernel
    int datasize = 3*sizeof(int);
    int x_dim = 10;
    int y_dim = 10;
    
    config.size[0] = x_dim;
    config.size[1] = y_dim;
    config.element_size = datasize;

    config.num_threads = test_threads;
    config.num_blocks = 1;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    
    testModel1(stream, d_rng_state, config); // Tested and working
    

    return 0;
}