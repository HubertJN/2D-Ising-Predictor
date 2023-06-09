#include "../include/model_wrappers.h"

int init_model(ising_model_config launch_struct) {
    // Add model specific launch parameters
    switch(launch_struct -> model_id) {
        case 1:
            launch_struct.element_size = 3*sizeof(int);
            break;
        case 2:
            launch_struct.element_size = 3*sizeof(int);
            break;
        default:
            fprintf(stderr, "Invalid model selection.\n");
            break;
    }
}

int launch_mc_sweep(cudaStream_t stream, curandState *state, ising_model_config launch_struct, int *device_array) {
    /*
      * This launches the original model. Single thread per grid stored in shared memory.
      *
      * Updates to this should not let the function block it should add tasks to the stream.
      *
      * Firstly transfer any initial grid configuration to the device.
      * Then launch the kernel.
      * Then transfer the result back to the host.
      * 
      * Parameters:
      *    stream: cuda stream to use
      *    state: curandState array to use
      *    launch_struct: struct containing launch parameters
    */

    // Transfer initial grid configuration to device
    if (launch_struct.initial_grid != NULL) {
        // create pinned host memory
        int *host_array;
        gpuErrchk( cudaMallocHost((void **)&host_array, launch_struct.element_size * launch_struct.size[0] * launch_struct.size[1]) );
        // If we have initial grid(s) to load, load them, and transfer it to the device 
        load_grid(launch_struct, host_array) 
    } 
    else {
        // If we don't have an initial grid to load, generate one
        
        // TODO: Implement this via a kernal launch
    }

    //TODO: Create d_Pacc and d_neighbour list here and refactor precomputations to be flexible
    // Allocate memory for d_Pacc and d_neighbour_list
    __consatant__ float d_Pacc[20];
    __consatant__ int d_neighbour_list[launch_struct.size[0] * launch_struct.size[1] * 4];

    // Precompute
    preComputeProbs(launch_struct, d_Pacc);
    preComputeNeighbours(launch_struct, d_neighbour_list);
    // Launch kernel
    mc_sweep<<<launch_struct.num_blocks, launch_struct.num_concurrent, 0, stream>>>(state, device_array, launch_struct.size[0], launch_struct.num_concurrent);
}


// 
int testModel1(cudaStream_t stream, curandState *state, ising_model_config launch_struct, int *device_array) {
    // This tests the kernal that uses one thread to fill its grid sequentially.

    // Launch kernel
    test_1<<<launch_struct.num_blocks, launch_struct.num_concurrent, 0, stream>>>(state, device_array, launch_struct.size[0], launch_struct.size[1], launch_struct.num_concurrent);

    // Collect result
    int *array = (int *)malloc(launch_struct.size[0]*launch_struct.size[1] * launch_struct.element_size);
    cudaMemcpy(array, device_array, launch_struct.element_size * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent, cudaMemcpyDeviceToHost);
    
    // Each element is 3 ints so we multiply by 3 and add commas and newlines appropiately
    for(int i=0; i<launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent * 3; i++) {
        if (i % 3 == 0) {
            fprintf(stdout, ", ");
        }
        if(i % (launch_struct.size[0] * 3) == 0) {
            fprintf(stdout, "\n");
            if (i % (launch_struct.size[0] * 3 * launch_struct.size[1]) == 0)
            {
                fprintf(stdout, "\n");
            }
        }
        fprintf(stdout, "%d", array[i]);
    }
    fprintf(stdout, "\n");
    return 0;
}

