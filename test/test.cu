#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#include "../include/kernels.h"
#include "../include/model_wrappers.h"

int main( int argc, char *argv[]) {

    // Single Kernel test.

    printf("Single Kernel Test\n");

    //Misc Vars
    int test_threads = 5;

    // Initialize RNG
    curandState_t *d_rng_state;
    // Allocate each thread memory for a random state 
    cudaMalloc((void **)&d_rng_state, test_threads * sizeof(curandState)); 

    init_rng<<<1, test_threads>>>(d_rng_state, time(NULL), test_threads);

    // run the test kernel
    int datasize = 3;
    int x = 10;
    int y = 10;
    int x_dim = x*datasize;
    int y_dim = y;
    fprintf(stdout, "%d, %d", x_dim, y_dim);
    int *array;
    cudaMalloc((void **)&array, x_dim*y_dim*test_threads * sizeof(int));

    test_1<<<1, test_threads>>>(d_rng_state, array, x_dim, y_dim, test_threads);

    cudaDeviceSynchronize();

    int *host_array = (int *)malloc(x_dim*y_dim*test_threads * sizeof(int));
    cudaMemcpy(host_array, array, x_dim*y_dim*test_threads * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for(int i=0; i<x_dim * y_dim * test_threads; i++) 
    {
        if(i % datasize == 0) {
            fprintf(stdout, ",");
        }
        if(i % x_dim == 0) {
            fprintf(stdout, "\n");
            if (i % (x_dim * y_dim) == 0)
            {
                fprintf(stdout, "\n");
            }
        }
        fprintf(stdout, "%d", host_array[i]);
    } 
    fprintf(stdout, "\n");
    return 0;
}