#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <pthread.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 

#include "../include/input_reader.h"
#include "../include/model_wrappers.h"
#include "../include/shared_data.h"

int main(int argc, char *argv[]) {

    // Low quality randomness only used for file identifiers
    srand (time(NULL));

    // Read the configuration file ===============================================
    char* filename;
    const char* callpath;
    // Check if command line arguments were provided
    printf("Number of arguments: %d\n", argc);
    if (argc == 2) {
        callpath = argv[0];
        filename = argv[1];
        
        printf("Callpath: %s\n", callpath);
        printf("Filename: %s\n", filename);
    }
    else {
        printf("Usage: %s [filename] \n", argv[0]);
        exit(1);
    }
    int models = 0;
    get_number_of_models(filename, &models);
    // Allocate memory for the array of pointers to structs
    ising_model_config* params_array[models];
    if (params_array == NULL) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        exit(1);
    }
    read_input_file(filename, params_array, models, callpath);
    fprintf(stderr, "Number of models: %d\n", sizeof(params_array)/sizeof(params_array[0]));
    
    // Print the models we are going to run
    for (int i = 0; i < models; i++) {
        init_model(params_array[i]);
        // Print the parameters of the model
        fprintf(stderr, "Model #: %d, ModelID: %d\n", i, params_array[i]->model_id);
        fprintf(stderr, "Grid x,y, %d, %d\n", params_array[i]->size[0], params_array[i]->size[1]);
        fprintf(stderr, "Number of iterations: %d\n", params_array[i]->iterations);
        fprintf(stderr, "Number of iterations per step: %d\n", params_array[i]->iter_per_step);
        fprintf(stderr, "Number of concurrent models: %d\n", params_array[i]->num_concurrent);
        fprintf(stderr, "Temprature: %f\n", params_array[i]->inv_temperature);
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
    int L2_cache_size = deviceProp.l2CacheSize;
    int num_multiprocessors = deviceProp.multiProcessorCount;

    // Print device specs
    fprintf(stderr, "Device name: %s\n", deviceProp.name);
    fprintf(stderr, "Max threads per block: %d\n", max_threads_per_block);
    fprintf(stderr, "Max shared memory per block: %d\n", max_shared_memory_per_block);


    size_t mem_size;
    int nBlocks;
    int nThreads;
    int min_blocks;
    int max_blocks;
    int prob_size;
    int nStreams = 0;
    int total_blocks = 0;
    int total_threads = 0;
    int cache_limit;
    // Loop over all concurrent models
    for (int i = 0; i < models; i++) {
        nBlocks = 0;
        nThreads = 0;
        // Check the concurrency style of the model and set the number of blocks and threads
        switch (params_array[i]->model_id)
        {
        case 1:
            mem_size = params_array[i] -> size[0] * params_array[i] -> size[1] * params_array[i] -> element_size;
            params_array[i] -> mem_size = mem_size;
            // Cache Limit is the number of grids that can fit the the L2 cache
            cache_limit =  L2_cache_size / params_array[i]->mem_size;
            //Concurrency style 1: Models are stored in device memory
            nThreads = params_array[i]->num_concurrent;

            // We want a minimum of 128 threads per block i.e. 4 warps and a maximum of 1024 threads per block i.e. 32 warps
            // We want to launch N*SM blocks per model

            min_blocks = (nThreads + max_threads_per_block - 1) / max_threads_per_block;
            max_blocks = (nThreads + 128 - 1) / 128;

            if (num_multiprocessors > max_blocks) {
                nBlocks = max_blocks;
                nThreads = 128;
            }
            else {
                nBlocks = (nThreads/128) / (num_multiprocessors * cache_limit);
                if (nBlocks < min_blocks) {
                    nBlocks = min_blocks;
                    nThreads = max_threads_per_block;
                }
                else {
                    nThreads = 128*cache_limit*num_multiprocessors;
                }
            }


            if (nThreads > max_threads_per_block) {
                fprintf(stderr, "Error: Too many threads per block for model %d\n", i);
                exit(1);
            }
            prob_size = 10; // The length of the precomputed proabibilities array
            break;

        case 2:
            // Concurrency style 2: Each model fills a block's shared memory
            nBlocks = params_array[i]->num_concurrent;
            nThreads = nBlocks * params_array[i]->num_threads;
            break;
        
        default:
            break;
        }

        // Apply the computed values to the model parameters for kernel launch
        params_array[i]->num_blocks = nBlocks;
        params_array[i]->num_threads = nThreads;
        params_array[i]->prob_size = prob_size;

        // Update the total number of blocks and threads
        total_blocks += nBlocks;
        total_threads += nThreads*nBlocks;
        // for each model launch add a stream
        nStreams++;
    }

    // Check the number of threads and blocks are supported by the device
        // Check if there is a sensible stream queue that can be used to make
        // the most of the device

    // Arguably this should max out at the number of blocks that can be run on the device
    int random_blocks = (total_threads + max_threads_per_block - 1) / max_threads_per_block;
    int random_threads = total_threads/random_blocks;
   

    
    curandState *dev_states;
    // Give each potential thread a random state
    cudaMalloc((void **)&dev_states, total_threads * sizeof(curandState));
    
    init_rng<<<random_blocks, random_threads>>>(dev_states, time(NULL), total_threads);
    cudaDeviceSynchronize();
    // ============================================================================

    // Create streams and allocate memory for grids on device =====================

    // Create CUDA streams based on the model configurations
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // Loop over the models, load grids where required, and allocate memory for the grids on the device
    int* h_data[nStreams];
    int* d_data[nStreams];
    // LOOP
    for (int i=0; i < nStreams; i++) {
        mem_size = params_array[i] -> mem_size * params_array[i] -> num_concurrent;
        params_array[i] -> mem_size = mem_size;
        // Allocate required space on host and device
        cudaMalloc(&d_data[i], mem_size);
        h_data[i] = (int *)malloc(mem_size);
    }
    //LOOP END
    // ============================================================================

    // Threaded CPU code ==========================================================
    pthread_t threads[nStreams];
    // Create an array of pointers to structs to pass to the threads
    struct model_thread_data* thread_structs[nStreams];

    // Queue memcopys and CUDA kernels on multiple streams ========================
    for (int i=0; i < nStreams; i++) {
        //create a thread for each model
        // Switch to select the correct launch wrapper
        switch(params_array[i] -> model_id) {
            case 1:
                thread_structs[i] = (model_thread_data*)malloc(sizeof(model_thread_data));
                thread_structs[i] -> stream = stream[i];
                thread_structs[i] -> dev_states = dev_states;
                thread_structs[i] -> params_array = params_array[i];
                thread_structs[i] -> h_data = h_data[i];
                thread_structs[i] -> d_data = d_data[i];
                thread_structs[i] -> idx = i;
                pthread_create(&threads[i], NULL, launch_mc_sweep, thread_structs[i]);
                break;
            default:
                fprintf(stderr, "Invalid model selection.\n");
                break;
        }
    }
    // ============================================================================
    // Wait for all threads to finish =============================================
    for (int i=0; i < nStreams; i++) {
        pthread_join(threads[i], NULL);
    }
    // End of threaded CPU code ===================================================

    // Copy results back to host ==================================================
    for (int i=0; i < nStreams; i++) {
        // TODO: make this async
        switch (params_array[i] -> model_id)
        {
            case 1:
                cudaMemcpy(h_data[i], d_data[i], params_array[i] -> mem_size, cudaMemcpyDeviceToHost);
                break;
            default:
                fprintf(stderr, "Invalid model selection.\n");
                break;
        }
    }

    // ============================================================================

    // Print results ==============================================================
    for (int i=0; i < nStreams; i++) {
        
    }


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
