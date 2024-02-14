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

void* launch_mc_sweep(void *arg) {
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

    // Unpack cpu thread struct
    struct model_thread_data *thread_data = (struct model_thread_data*)arg;
    cudaStream_t stream = thread_data->stream;
    curandState *state = thread_data->dev_states;
    ising_model_config* launch_struct = thread_data->params_array;
    int *host_array = thread_data->h_data;
    int *device_array = thread_data->d_data;
    int stream_ix = thread_data->idx;

    // Allocate memory for device array
    //cudaMalloc((void **)&device_array, launch_struct->element_size * launch_struct->size[0] * launch_struct->size[1]);

    switch(launch_struct->starting_config) {
        case 0:
            // Load Grid from file
            if (launch_struct->input_file != NULL) {
                // If we have initial grid(s) to load, load them, and transfer it to the device 
                load_grid(stream, launch_struct, host_array, device_array);
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
            } 
            else {
                fprintf(stderr, "No initial grid to load.\n");
            }
            break;
        case 1:
            // Random
            init_rand_grids<<<launch_struct->num_blocks, launch_struct->num_threads, 0, stream>>>(state, launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            break;
        case 2:
            // All up
            fprintf(stderr, "All up\n");
            init_ud_grids<<<launch_struct->num_blocks, launch_struct->num_threads, 0, stream>>>(launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, 1);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            fprintf(stderr, "All are up\n");
            break;
        case 3:
            // All down
            init_ud_grids<<<launch_struct->num_blocks, launch_struct->num_threads, 0, stream>>>(launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, -1);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            break;
        default:
            fprintf(stderr, "Invalid starting configuration.\n");
            break;
    }

    // Get initial grid from device
    gpuErrchk( cudaMemcpy(host_array, device_array, launch_struct->mem_size, cudaMemcpyDeviceToHost) );

    //TODO: Create d_Pacc and d_neighbour list here and refactor precomputations to be flexible
    // Allocate memory for d_Pacc and d_neighbour_list
    int prob_size = 10;
    launch_struct->prob_size = prob_size;

    // Allocate device memory for d_Pacc and d_neighbour_list (there is potential here to put this in a faster memory location?)
    float* d_Pacc;
    int* d_neighbour_list;
    cudaMalloc(&d_Pacc, prob_size * sizeof(float));
    cudaMalloc(&d_neighbour_list, launch_struct->size[0] * launch_struct->size[1] * 4 * sizeof(int));
    fprintf(stderr, "Allocated memory for d_Pacc and d_neighbour_list\n");

    // Precompute
    preComputeProbs(launch_struct, d_Pacc);
    preComputeNeighbours(launch_struct, d_neighbour_list);
    fprintf(stderr, "Precomputed probs and neighbours\n");

    // Allocate memory for magnetisation and energy
    float h_magnetisation[launch_struct->num_concurrent];
    float* d_magnetisation;
    cudaMalloc(&d_magnetisation, launch_struct->num_concurrent * sizeof(float));
    float h_nucleation[launch_struct->num_concurrent];
    // Pass threshold values to device
    float* d_up_threshold;
    float* d_dn_threshold;
    cudaMalloc(&d_up_threshold, sizeof(float));
    cudaMalloc(&d_dn_threshold, sizeof(float));
    // Copy to device
    gpuErrchk(cudaMemcpy(d_up_threshold, &launch_struct->up_threshold, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_dn_threshold, &launch_struct->dn_threshold, sizeof(float), cudaMemcpyHostToDevice));
    // Copy task to device
    int* d_model_itask;
    cudaMalloc(&d_model_itask, sizeof(int));
    gpuErrchk(cudaMemcpy(d_model_itask, &launch_struct->model_itask, sizeof(int), cudaMemcpyHostToDevice));

    // Start output file for this model
    // Uid for model, one file for all concurrent copies
    file_handle theHdl;
    std::fstream metafile;
    char filename[PATH_MAX];
    fillCompletePath(filename);
    pthread_mutex_lock(&metafile_lock);
    fprintf(stderr, "Writing model metadata");
    snprintf(filename+strlen(filename), sizeof(filename)-strlen(filename), "/grid.meta");
    metafile.open(filename, std::ios::out | std::ios::app);
    outputInitialInfo(theHdl, launch_struct, stream_ix);
    outputModelId(metafile, theHdl, launch_struct);
    metafile.close();
    pthread_mutex_unlock(&metafile_lock);
    // Launch kernel
    for (int i = 0; i < launch_struct->iterations; i+=launch_struct->iter_per_step){
        mc_sweep<<<launch_struct->num_blocks, launch_struct->num_threads, 0, stream>>>(state, launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, launch_struct->inv_temperature, launch_struct->field, launch_struct->iter_per_step, d_neighbour_list, d_Pacc);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(stream) );
        gpuErrchk( cudaMemcpy(host_array, device_array, launch_struct->mem_size, cudaMemcpyDeviceToHost));
        fprintf(stderr, "Iterations %d to %d\n", i, i+launch_struct->iter_per_step);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        // Compute energy and magnetisation (GPU)
        compute_magnetisation<<<launch_struct->num_blocks, launch_struct->num_threads, 0, stream>>>(launch_struct->size[0], launch_struct->size[1], launch_struct->num_concurrent, device_array, d_magnetisation);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaStreamSynchronize(stream) );
        gpuErrchk( cudaMemcpy(h_magnetisation, d_magnetisation, launch_struct->num_concurrent * sizeof(float), cudaMemcpyDeviceToHost));
        gpuErrchk( cudaPeekAtLastError() );
        // Write to file (CPU)
        writeAllGrids(theHdl, host_array, i,  stream_ix);
        // Check for full nucleation or resolved fates
        int full_nucleation = 0;
        int fate_down = 0;
        for (int j = 0; j < launch_struct->num_concurrent; j++) {
            if (h_magnetisation[j] > launch_struct->up_threshold) {
                fprintf(stderr, "Nucleation detected at iteration %d on grid %d\n", i, j+1);
                full_nucleation++;
            }
            else if (launch_struct->model_itask == 0 && h_magnetisation[j] < launch_struct->dn_threshold) {
                fprintf(stderr, "Resolved fate detected at iteration %d on grid %d\n", i, j+1);
                full_nucleation++;
                fate_down++;
            }
        }
        int fate_up = full_nucleation - fate_down;
        int fraction_up = (float(fate_up)) / float(full_nucleation);
        if (launch_struct->model_itask == 0) {
            printf("\r Sweep : %10d, Reached m = %6.2f : %4d , Reached m = %6.2f : %4d , Unresolved : %4d, pB = %10.6f",
            i, launch_struct->dn_threshold, fate_down, launch_struct->up_threshold, fate_up, launch_struct->num_concurrent-full_nucleation, fraction_up);
        }

        if (full_nucleation == launch_struct->num_concurrent) {
            if (launch_struct->model_itask == 0) {
                fprintf(stderr, "Resolved fate on all grids \n");
            } else {
                fprintf(stderr, "Full nucleation on all grids \n");
            }
            break;
        }
    }

    finaliseFile(theHdl);

   return;
}
