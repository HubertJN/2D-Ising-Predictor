#include "../include/model_wrappers.h"

//Todo this should go to helpers
void init_model(ising_model_config* launch_struct) {
    // Add model specific launch parameters
    switch(launch_struct->model_id) {
        case 1:
            launch_struct->element_size = sizeof(int);
            break;
        default:
            fprintf(stderr, "Invalid model selection.\n");
            break;
    }
    return;
}

void launch_mc_sweep(cudaStream_t stream, curandState *state, ising_model_config* launch_struct, int *device_array) {
    /*
      * This launches the original model. Single thread per grid.
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

    // Allocate memory for device array
    //cudaMalloc((void **)&device_array, launch_struct->element_size * launch_struct->size[0] * launch_struct->size[1]);

    switch(launch_struct->starting_config) {
        case 0:
            // Load Grid from file
            if (launch_struct->input_file != NULL) {
                // If we have initial grid(s) to load, load them, and transfer it to the device 
                load_grid(stream, launch_struct, device_array);
            } 
            else {
                fprintf(stderr, "No initial grid to load.\n");
            }
            break;
        case 1:
            // Random
            init_rand_grids<<<launch_struct->num_blocks, launch_struct->num_concurrent, 0, stream>>>(state, launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array);
            break;
        case 2:
            // All up
            init_ud_grids<<<launch_struct->num_blocks, launch_struct->num_concurrent, 0, stream>>>(launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, 1);
            break;
        case 3:
            // All down
            init_ud_grids<<<launch_struct->num_blocks, launch_struct->num_concurrent, 0, stream>>>(launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, -1);
            break;
        default:
            fprintf(stderr, "Invalid starting configuration.\n");
            break;
    }

    //TODO: Create d_Pacc and d_neighbour list here and refactor precomputations to be flexible
    // Allocate memory for d_Pacc and d_neighbour_list
    int prob_size = 10;
    launch_struct->prob_size = prob_size;

    // Allocate device memory for d_Pacc and d_neighbour_list (there is potential here to put this in a faster memory location?)
    float* d_Pacc;
    int* d_neighbour_list;
    cudaMalloc(&d_Pacc, prob_size * sizeof(float));
    cudaMalloc(&d_neighbour_list, launch_struct->size[0] * launch_struct->size[1] * 4 * sizeof(int));

    // Precompute
    preComputeProbs(launch_struct, d_Pacc);
    preComputeNeighbours(launch_struct, d_neighbour_list);
    // Launch kernel
    mc_sweep<<<launch_struct->num_blocks, launch_struct->num_concurrent, 0, stream>>>(state, launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, launch_struct->inv_temperature, launch_struct->field, launch_struct->iter_per_step, d_neighbour_list, d_Pacc);

    return;
}
