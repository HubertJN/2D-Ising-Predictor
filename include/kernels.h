#ifndef KERNELS_H
#define KERNELS_H


#include <curand_kernel.h>
#include <float.h>

__global__ void init_rng(curandState *state, unsigned int seed, int n);
__global__ void compute_magnetisation(const int L_x, const int L_y, const int ngrids, const float nuc_threshold, int *device_array, float *magnetisation, int *up_threshold, int *dn_threshold, int *model_itask);

__global__ void init_rand_grids(curandState *state, const int L_x, const int L_y, const int ngrids, int *device_grids);
__global__ void init_ud_grids(const int L_x, const int L_y, const int ngrids, int *device_grids, const int u_d);

__global__ void mc_sweep(curandState *state, const int L_x, const int L_y, const int ngrids, int *d_ising_grids, const float beta, const float h, int nsweeps, int *d_neighbour_list, float *d_Pacc);

#endif // KERNELS_H