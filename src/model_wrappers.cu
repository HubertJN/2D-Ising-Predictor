#include "../include/model_wrappers.h"

// 
void testModel1(cudaStream_t stream, curandState *state, ising_model_config launch_struct) {
    // This tests the kernal that uses one thread to fill its grid sequentially.
    // Create pointers to device memory
    int *device_array;
    // Allocate device memory
    cudaMalloc((void **) &device_array, sizeof(float) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);

    // Launch kernel
    test_1<<<launch_struct.num_concurrent, 1, 0, stream>>>(state, device_array, launch_struct.size[0], launch_struct.size[1], launch_struct.num_concurrent);

    // Collect result
    float *array;
    cudaMemcpy((void**)&array, (void**)&device_array, sizeof(float) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent, cudaMemcpyDeviceToHost);
    
    // Print result (TODO: to file)
    for(int i=0; i<launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent; i++) {
        if(i % launch_struct.size[0] == 0) {
            fprintf(stdout, "\n");
        }
        if (i % (launch_struct.size[0] * launch_struct.size[1]) == 0)
        {
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "%f ", array[i]);
    }
}

void testModel2(cudaStream_t stream, curandState *state, ising_model_config launch_struct) {
    // This tests the kernal tht uses mutiple threads to fill its grid concurrently.
    int *device_array;
    // Allocate device memory
    cudaMalloc((void **) &device_array, sizeof(float) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);

    // Launch kernel
    test_2<<<launch_struct.num_threads, launch_struct.num_concurrent, 0, stream>>>(state, device_array, launch_struct.num_threads, launch_struct.size[0], launch_struct.size[1], launch_struct.num_concurrent);

    // Collect result
    float *array;
    cudaMemcpy((void**)&array, (void**)&device_array, sizeof(float) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent, cudaMemcpyDeviceToHost);

    // Print result (TODO: to file)
    for(int i=0; i<launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent; i++) 
    {
        if(i % launch_struct.size[0] == 0) {
            fprintf(stdout, "\n");
            if (i % (launch_struct.size[0] * launch_struct.size[1]) == 0)
            {
                fprintf(stdout, "\n");
            }
        }
        fprintf(stdout, "%f ", array[i]);
    } 
    

}


void launchModel1(cudaStream_t stream, curandState *state, ising_model_config launch_struct) {
    // This model launches a kernel that is fully initilised on host working in shared memory. 
    // This function initilises the device memory and launches the kernel, then collects the result and frees the device memory.
    // The kernel is defined in kernels.h
    // The kernel is launched on the stream passed as argument.

    // Create pointers to device memory
    float *device_grid;
    // Allocate device memory
    cudaMalloc((void **) &device_grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);
    // Allocate pinned host memory
    int *grid;
    cudaMallocHost((void **) &grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);
    int *magnetisation;
    cudaMallocHost((void **) &magnetisation, sizeof(int) * launch_struct.num_concurrent);

    for(int i=0; i<launch_struct.iterations; i+=launch_struct.iter_per_step) {
        // Launch kernel
        // ising_kernel_many<<<launch_struct.num_concurrent, 1, 0, stream>>>(state, device_grid, launch_struct.size[0], launch_struct.size[1], launch_struct.iter_per_step, launch_struct.temperature);
        // compute_magnetisation<<<launch_struct.num_concurrent, 1, 0, stream>>>(device_grid, launch_struct.size[0], launch_struct.size[1], magnetisation);
        
        // Collect result
        cudaMemcpyAsync(grid, device_grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1], cudaMemcpyDeviceToHost, stream);
    }

    // Free device memory
    cudaFree(device_grid);
    // Free pinned host memory
    cudaFreeHost(grid);
    cudaFreeHost(magnetisation);
}

void launchModel2(cudaStream_t stream, curandState *state, ising_model_config launch_struct) {
    // This model launches a kernel that is fully initilised on host working in shared memory. 
    // This function initilises the device memory and launches the kernel, then collects the result and frees the device memory.
    // The kernel is defined in kernels.h
    // The kernel is launched on the stream passed as argument.

    // Create pointers to device memory
    float *device_grid;
    // Allocate device memory
    cudaMalloc((void **) &device_grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);
    // Allocate pinned host memory
    int *grid;
    cudaMallocHost((void **) &grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);
    int *magnetisation;
    cudaMallocHost((void **) &magnetisation, sizeof(int) * launch_struct.num_concurrent);

    for(int i=0; i<launch_struct.iterations; i+=launch_struct.iter_per_step) {
        // Launch kernel
        // ising_kernel_many<<<launch_struct.num_concurrent, 1, 0, stream>>>(state, device_grid, launch_struct.size[0], launch_struct.size[1], launch_struct.iter_per_step, launch_struct.temperature);
        // compute_magnetisation<<<launch_struct.num_concurrent, 1, 0, stream>>>(device_grid, launch_struct.size[0], launch_struct.size[1], magnetisation);
        
        // Collect result
        cudaMemcpyAsync(grid, device_grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1], cudaMemcpyDeviceToHost, stream);
    }

    // Free device memory
    cudaFree(device_grid);
    // Free pinned host memory
    cudaFreeHost(grid);
    cudaFreeHost(magnetisation);
}

void launchModel3(cudaStream_t stream, curandState *state, ising_model_config launch_struct) {
    // This model launches a kernel that is fully initilised on host working in shared memory. 
    // This function initilises the device memory and launches the kernel, then collects the result and frees the device memory.
    // The kernel is defined in kernels.h
    // The kernel is launched on the stream passed as argument.

    // Create pointers to device memory
    float *device_grid;
    // Allocate device memory
    cudaMalloc((void **) &device_grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);
    // Allocate pinned host memory
    int *grid;
    cudaMallocHost((void **) &grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1] * launch_struct.num_concurrent);
    int *magnetisation;
    cudaMallocHost((void **) &magnetisation, sizeof(int) * launch_struct.num_concurrent);

    for(int i=0; i<launch_struct.iterations; i+=launch_struct.iter_per_step) {
        // Launch kernel
        // ising_kernel_many<<<launch_struct.num_concurrent, 1, 0, stream>>>(state, device_grid, launch_struct.size[0], launch_struct.size[1], launch_struct.iter_per_step, launch_struct.temperature);
        // compute_magnetisation<<<launch_struct.num_concurrent, 1, 0, stream>>>(device_grid, launch_struct.size[0], launch_struct.size[1], magnetisation);
        
        // Collect result
        cudaMemcpyAsync(grid, device_grid, sizeof(int) * launch_struct.size[0] * launch_struct.size[1], cudaMemcpyDeviceToHost, stream);
    }

    // Free device memory
    cudaFree(device_grid);
    // Free pinned host memory
    cudaFreeHost(grid);
    cudaFreeHost(magnetisation);
}