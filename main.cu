#include <cuda.h>

#include "input_reader.h"

#define N_STATES (N_THREADS * N_BLOCKS)

int main() {

    // Read the configuration file ===============================================

    //default filename
    char* filename = "input.txt";
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

    // This should map to the number of models
    cudaMalloc((void **)&dev_states, N_STATES * sizeof(curandState));

    init_rng<<<N_BLOCKS, N_THREADS>>>(dev_states, time(NULL), N_STATES);
    cudaDeviceSynchronize();
    // ============================================================================

    // Run compatibility checks between selected device and model configurations ==

    // TODO: Write those checks

    // ============================================================================

    // Create streams and allocate memory for grids on device =====================

    // Create CUDA streams
    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // Allocate memory on the CUDA device
    float *d_data[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaMalloc(&d_data[i], stream_size * sizeof(float));
    }

    // Pin memory on the host if required
    // TODO: Write this

    // ============================================================================

    // Queue memcopys and CUDA kernels on multiple streams ========================
    for (int i = 0; i < num_streams; i++) {
        // Copy the model configuration to the device
        // TODO: Write this
        // Copy the grid to the device, TODO: add switch for different grid types
        cudaMemcpyAsync(&d_data[i][0], &stream_size, sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        // Launch the CUDA kernel
        switch(model_in) {
            case 1:
                launchModel1(stream[i], dev_states, launch_struct_ptr);
                break;
            case 2:
                launchModel2(stream[i], dev_states, launch_struct_ptr);
                break;
            case 3:
                launchModel3(stream[i], dev_states, launch_struct_ptr);
                break;
            default:
                printf("Invalid model selection.\n");
                break;
        }
        // Copy the grid back to the host TODO: add switch for different grid types?
        cudaMemcpyAsync(&h_data[i][0], &d_data[i][0], stream_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
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
