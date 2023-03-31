#include <cuda.h>
#define N_STATES (N_THREADS * N_BLOCKS)

int main() {
    int num_streams = 4; // Number of CUDA streams. This should be set via a makefile configiuration.
    

    // Initialize CUDA
    cudaSetDevice(0);
    curandState *dev_states;

    cudaMalloc((void **)&dev_states, N_STATES * sizeof(curandState));

    init_rng<<<N_BLOCKS, N_THREADS>>>(dev_states, time(NULL), N_STATES);


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

    // Launch CUDA kernels on multiple streams
    for (int i = 0; i < num_streams; i++) {
        cudaMemcpyAsync(&d_data[i][0], &stream_size, sizeof(float), cudaMemcpyHostToDevice, stream[i]);
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
        cudaMemcpyAsync(&h_data[i][0], &d_data[i][0], stream_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }

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

    return 0;
}
