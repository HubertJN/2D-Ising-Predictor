#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "../include/input_reader.h"
#include "../include/model_wrappers.h"

int main(int argc, char *argv[]) {

    // Read the configuration file ===============================================
    char* filename;
    // Check if command line arguments were provided
    if (argc > 1) {
        filename = argv[1];
    }
    // pattern for further arguments
    // if (argc > 2) {
    //     count = atoi(argv[2]);
    // } 
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [filename] \n", argv[0]);
        exit(1);
    }

    // Read the number of models from the input file and set the number of streams
    int num_streams;
    get_number_of_models(filename, &num_streams);


    // Allocate memory for the array of pointers to structs containing model configurations
    ising_model_config* params_array[num_streams];
    if (params_array == NULL) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        exit(1);
    }
    read_input_file(filename, params_array, num_streams);

    // Print the number of models
    fprintf(stderr, "Number of models: %d\n", sizeof(params_array)/sizeof(params_array[0]));
    // Configuration loaded, now run the parallel simulations in multiple streams.

    // ============================================================================


    // Initialize CUDA ===========================================================

    // Get device information
    // TODO: Put in the getinfo command

    // Set the device to use
    cudaSetDevice(0);
    curandState *dev_states;

    // Run compatibility checks between selected device and model configurations

    // TODO: Write those checks

    // use the checks to set the number of blocks and threads

    // set to 1 for now, in the future this should be set to the number of of blocks needed to run the model
    int n_blocks = 1;
    int n_threads;
    for (int i = 0; i < num_streams; i++) {
        // for now this is just the number of concurrent threads, in the future this should be the number of threads needed to run the model
        n_threads += params_array[i] -> num_concurrent;
    }
    

    // This should map to the number of models
    cudaMalloc((void **)&dev_states, n_blocks * n_threads * sizeof(curandState));

    init_rng<<<n_blocks, n_threads>>>(dev_states, time(NULL), num_streams);
    cudaDeviceSynchronize();
    // ============================================================================



    // Create streams and allocate memory for grids on device =====================

    // Create CUDA streams
    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // Allocate memory on the CUDA device
    int stream_size; 
    float h_data[num_streams];
    float *d_data[num_streams];
    for (int i = 0; i < num_streams; i++) {
        stream_size = params_array[0] -> size[0] * params_array[0] -> size[0];
        cudaMalloc(&d_data[i], stream_size * sizeof(int));
    }

    // Pin memory on the host if required
    // TODO: Write this

    // ============================================================================

    // Queue memcopys and CUDA kernels on multiple streams ========================
    for (int i = 0; i < num_streams; i++) {
        // Copy the model configuration to the device
        // TODO: Write this
        // Copy the grid to the device, TODO: add switch for different grid types
        cudaMemcpyAsync(d_data[i], &h_data[i], sizeof(float) * sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        // Launch the CUDA kernel
        switch(params_array[i] -> model_id) {
            case 1:
                launchModel1(stream[i], dev_states, *params_array[i]);
                break;
            case 2:
                launchModel2(stream[i], dev_states, *params_array[i]);
                break;
            case 3:
                launchModel3(stream[i], dev_states, *params_array[i]);
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
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(stream[i]);
    }

    // Destroy CUDA streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }

    // Free allocated memory on the CUDA device
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_data[i]);
    }

    // Free the RNG on the device
    cudaFree(dev_states);

    // TODO: Free other memory?

    // ============================================================================

    return 0;
}
