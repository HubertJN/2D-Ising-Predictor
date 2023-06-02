#ifndef KERNELS_H
#define KERNELS_H

#include <curand_kernel.h>

__global__ void init_rng(curandState *state, unsigned int seed, int n);
__global__ void compute_magnetisation(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation);

__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);



#endif // KERNELS_H
