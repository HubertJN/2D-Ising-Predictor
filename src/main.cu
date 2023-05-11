#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "../include/input_reader.h"
#include "../include/model_wrappers.h"

int main(int argc, char *argv[]) {

    // Read the configuration file ===============================================
    char* filename;// = "input.txt";
    // Check if command line arguments were provided
    if (argc > 1) {
        filename = argv[1];
    }
    // pattern for further arguments
    // if (argc > 2) {
    //     count = atoi(argv[2]);
    // } 
    if (argc > 2) {
        printf("Usage: %s [filename] \n", argv[0]);
        exit(1);
    }
    int models;
    get_number_of_models(filename, &models);
    // Allocate memory for the array of pointers to structs
    ising_model_config* params_array[models];
    if (params_array == NULL) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        exit(1);
    }
    read_input_file(filename, params_array, models);
    fprintf(stderr, "Number of models: %d\n", sizeof(params_array)/sizeof(params_array[0]));
    
    // Print the models we are going to run
    for (int i = 0; i < models; i++) {
        // Print the parameters of the model
        fprintf(stderr, "Model #: %d, ModelID: %d\n", i, params_array[i]->model_id);
        fprintf(stderr, "Grid x,y, %d, %d\n", params_array[i]->size[0], params_array[i]->size[1]);
        fprintf(stderr, "Number of iterations: %d\n", params_array[i]->iterations);
        fprintf(stderr, "Number of iterations per step: %d\n", params_array[i]->iter_per_step);
        fprintf(stderr, "Number of concurrent models: %d\n", params_array[i]->num_concurrent);
        fprintf(stderr, "Temprature: %f\n", params_array[i]->temperature);
    }
    // Configuration loaded, now run the parallel simulations in multiple streams.

    // ============================================================================


    // Initialize CUDA ===========================================================

    // Get device information
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA capable devices found\n");
        exit(EXIT_FAILURE);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Set the device to use
    cudaSetDevice(0);

    // get specs of the device
    int max_threads_per_block = deviceProp.maxThreadsPerBlock;
    int max_shared_memory_per_block = deviceProp.sharedMemPerBlock;
    int max_blocks_per_multiprocessor = deviceProp.maxThreadsPerMultiProcessor / max_threads_per_block;
    int max_threads_per_multiprocessor = max_blocks_per_multiprocessor * max_threads_per_block;


    // Run compatibility checks between selected device and model configurations

    // TODO: Write those checks
    /*
    LOOP:
        Check for running mode overriden by user
            set user defined mode
            get number of threads supported by the model
            check user input is valid
                Check if the model can be run in shared memory
                Check if the model has a supported number of threads
                ...

        Check if multiple models can be run in a single block
            set concurrency style launch
        Check if the model can be run in block
            set full block style launch
            get number of threads supported by the model
        Check if the model can be run in multiple blocks
            set multi block style launch
            get number of threads supported by grids
        Check model can be run on device memory
            set device memory style launch
            get number of threads supported by the model
    END LOOP
    */

    // use the checks to set the number of blocks and threads
    
    /* Pseudo code for setting the number of blocks and threads
    total_blocks = 0
    total_threads = 0
    LOOP 
        Concurrancy style lauhches: 
            // Many models fit in a single blocks shared memory
            n_threads = number of concurrent models
            n_blocks = n_threads / max number of models per block
            n_streams ++

        Full block style launches:
            // Each model fills a blocks shared memory
            n_blocks = number of concurrent models
            n_threads = number of concurrent models * number of threads each model needs
            n_streams ++
        
        Multi block style launches:
        // Each model will not fit in a single blocks shared memory
            Style 1:
                // grid is split into multiple blocks
                n_blocks = grid size / block memory size
                n_threads = n_blocks * number of threads supported by each block
            Style 2:
                // grid is stored in device memory
                n_threads = number of threads supported by the model
                n_blocks = n_threads / max number of threads per block

            n_streams += concurrent models
            // multiply by the number of concurrent models
            
        
        // apply these to the model parameters to be used in the kernel launch
        // keep a running total of the number of blocks and threads for all launches
        total_blocks += n_blocks
        total_threads += n_threads
    END LOOP
    */
    int nBlocks;
    int nThreads;
    int nStreams = 0;
    int total_blocks = 0;
    int total_threads = 0;
    // Loop over all concurrent models
    for (int i = 0; i < models; i++) {
        nBlocks = 0;
        nThreads = 0;
        // Concurrency style 1: Many models fit in a single block's shared memory
        if (params_array[i]->model_id == 1) {
            nThreads = params_array[i]->num_concurrent;
            nBlocks = (nThreads + max_threads_per_block - 1) / max_threads_per_block;
        }

        if (params_array[i]->model_id == 2) {
            // Concurrency style 2: Each model fills a block's shared memory
            nBlocks = params_array[i]->num_concurrent;
            nThreads = nBlocks * params_array[i]->num_threads;
        }

        // Apply the computed values to the model parameters for kernel launch
        params_array[i]->num_blocks = nBlocks;
        params_array[i]->num_threads = nThreads;

        // Update the total number of blocks and threads
        total_blocks += nBlocks;
        total_threads += nThreads;
        nStreams++;
    }

    // Check the number of threads and blocks are supported by the device
        // Check if there is a sensible stream queue that can be used to make 
        // the most of the device

    int random_threads = total_threads;
    int random_blocks = (random_threads + max_threads_per_block - 1) / max_threads_per_block;
    
    curandState *dev_states;
    // Give each potential thread a random state
    cudaMalloc((void **)&dev_states, random_threads * sizeof(curandState));
    
    init_rng<<<random_blocks, random_threads>>>(dev_states, time(NULL), random_threads);
    cudaDeviceSynchronize();
    // ============================================================================



    // Create streams and allocate memory for grids on device =====================

    // Create CUDA streams based on the model configurations
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // Loop over the models, load grids where required, and allocate memory for the grids on the device
    int stream_size; 
    int h_data[nStreams];
    int *d_data[nStreams];
    // LOOP
        // Allocate memory on the CUDA device
        for (int i=0; i < nStreams; i++) {
            stream_size = params_array[i] -> size[0] * params_array[i] -> size[1];
            cudaMalloc(&d_data[i], stream_size * sizeof(int));
        }
        // Pin memory on the host if required
        // TODO: Write this
    //LOOP END
    // ============================================================================

    // Queue memcopys and CUDA kernels on multiple streams ========================
    for (int i=0; i < nStreams; i++) {
        // Copy the model configuration to the device
        // TODO: Write this
        // Copy the grid to the device, TODO: add switch for different grid types
        //cudaMemcpyAsync(d_data[i], &h_data[i], sizeof(float) * sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        // Launch the CUDA kernel
        switch(params_array[i] -> model_id) {
            case 1:
                testModel1(stream[i], dev_states, *params_array[i]);
                break;
            case 2:
                testModel2(stream[i], dev_states, *params_array[i]);
                break;
            default:
                fprintf(stderr, "Invalid model selection.\n");
                break;
        }
        // Copy the grid back to the host TODO: add switch for different grid types?
        cudaMemcpyAsync(&h_data[i], d_data[i], stream_size * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }
    // ============================================================================


    // Run cleanup ================================================================
    // Wait for CUDA streams to finish
    for (int i = 0; i < nStreams; i++) {
        cudaStreamSynchronize(stream[i]);
    }

    // Destroy CUDA streams
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(stream[i]);
    }

    // Free allocated memory on the CUDA device
    for (int i = 0; i < nStreams; i++) {
        cudaFree(d_data[i]);
    }

    // Free the RNG on the device
    cudaFree(dev_states);

    // TODO: Free other memory?

    // ============================================================================

    return 0;
}
