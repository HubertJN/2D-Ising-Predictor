#ifndef KERNELS_H
#define KERNELS_H

#include <curand_kernel.h>

__global__ void init_rng(curandState *state, unsigned int seed, int n);
__global__ void compute_magnetisation(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation);

#endif // KERNELS_H
