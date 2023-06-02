#include "../include/kernels.h"

__global__ void init_rng(curandState *state, unsigned int seed, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < n)
    {
        curand_init(seed, tid, 0, &state[tid]);
    }
}

// compute magnetisation on the gpu
__global__ void compute_magnetisation(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    int *loc_grid = &d_ising_grids[idx*L*L]; // pointer to device global memory

    float m = 0.0f;

    int i;
    for (i=0;i<L*L;i++) { m += loc_grid[i]; }
    d_magnetisation[idx] = m/(float)(L*L);

  }

  return;

}

// Concurrancy Style
__global__ void test_1(curandState *state, int *d_test_array, int size_x, int size_y, int concurrency) {
  // Allocate shared memory, potential crash/massive slowdown if not enough shared memory is available, this should be configured by the main program
  //extern __shared__ int shared_mem[];

  // Todo: copy loaded data into shared memory

  // Test kernal should set the values of its array to the thread index, block index and a random number

  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  // Calculate the array start point
  int start_idx = idx*size_x*3*size_y;
  // Set the values
  for (int i=0;i<size_x*size_y*3;i+=3) {
    if (i < size_x*size_y*3) {
      d_test_array[start_idx+i] = idx;
      d_test_array[start_idx+i+1] = blockIdx.x;
      d_test_array[start_idx+i+2] = (int)(curand_uniform(state)*100);
    }
  }
  return;
}

// Block style
__global__ void test_2(curandState *state, int *d_test_array, int nthreads, int size_x, int size_y, int concurrency) {
  // Allocate shared memory, potential crash/massive slowdown if not enough shared memory is available, this should be configured to use the whole block
  // extern __shared__ int shared_mem[];

  // Test kernal should set the values of its array to the thread index, block index and a random number
  int idx = threadIdx.x+blockIdx.x*blockDim.x;


  // Set the values
  for (int i=idx*3; i<size_x*size_y; i+=nthreads*3) {
    if (i < size_x*size_y) {
      d_test_array[i] = idx;
      d_test_array[i+1] = blockIdx.x;
      d_test_array[i+2] = (int)(curand_uniform(state)*100);
    }
  }
  return;
}