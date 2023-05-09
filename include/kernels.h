#ifndef KERNELS_H
#define KERNELS_H

#include <curand_kernel.h>

__global__ void init_rng(curandState *state, unsigned int seed, int n);
__global__ void compute_magnetisation(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation);

__global__ void test_1(curandState *state, int *d_test_array, int size_x, int size_y, int concurrency);
__global__ void test_2(curandState *state, int *d_test_array, int nthreads, int size_x, int size_y, int concurrency);

#endif // KERNELS_H
