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

// initialise the ising grids on the gpu
__global__ void init_rand_grids(curandState *state, const int L_x, const int L_y, const int ngrids, int *device_grids) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads
    curandState localState = state[idx];

    int N = L_x*L_y;
    // Avoid rounding errors after creating random numbers by ensuring out max
    // is upto 1.0f not up to 1.0f + FLT_EPSILON
    float shrink = (1.0f - FLT_EPSILON)*(float)N;

    for (int i=0;i<N;i++) {
      if (curand_uniform(&localState) < 0.5f) {
        device_grids[idx*N + i] = 1;
      } else {
        device_grids[idx*N + i] = -1;
      }
    }

  }

  return;
}

// initialise the ising grids on the gpu
__global__ void init_ud_grids(const int L_x, const int L_y, const int ngrids, int *device_grids, const int u_d) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    int N = L_x*L_y;

    for (int i=0;i<N;i++) {
      device_grids[idx*N + i] = u_d;
    }

  }

  return;
}

// sweep on the gpu - default version
__global__ void mc_sweep(curandState *state, const int L_x, const int L_y, const int ngrids, int *d_ising_grids, const float beta, const float h, int nsweeps, const int *d_neighbour_list, const float *d_Pacc) {
  /* 
    * Default version of the sweep kernel, uses a neighbour list to avoid branching.
    * 
    * Parameters:
    * state: pointer to the RNG state array
    * L_x: linear size of the grid (x dimension)
    * L_y: linear size of the grid (y dimension)
    * ngrids: number of grids
    * d_ising_grids: pointer to the array of grids
    * nsweeps: number of sweeps to perform
  */

  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int index;

  if (idx < ngrids) {

    // local copy of RNG state for current threads
    curandState localState = state[idx];

    int N = L_x*L_y;
    // Avoid rounding errors after creating random numbers by ensuring out max
    // is upto 1.0f not up to 1.0f + FLT_EPSILON
    float shrink = (1.0f - FLT_EPSILON)*(float)N;

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*N]; // pointer to device global memory 


    int imove, my_idx, spin, n1, n2, n3, n4; //row, col;
    for (imove=0;imove<N*nsweeps;imove++){

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));



      spin = loc_grid[my_idx];

      // find neighbours, periodic boundary conditions. D,U,L,R
      // row = my_idx/L_x;
      // col = my_idx%L_x;
      // n1 = loc_grid[L*((row+1)%L) + col];
      // n2 = loc_grid[L*((row+L-1)%L) + col];
      // n3 = loc_grid[L*row + (col+1)%L];
      // n4 = loc_grid[L*row + (col+L-1)%L];
      // use neighbour list to get neighbours from constant memory
      n1 = loc_grid[d_neighbour_list[my_idx] + 0];
      n2 = loc_grid[d_neighbour_list[my_idx] + 1];
      n3 = loc_grid[d_neighbour_list[my_idx] + 2];
      n4 = loc_grid[d_neighbour_list[my_idx] + 3];
      //n_sum = 4;
      // index = 5*(spin+1) + n1+n2+n3+n4 + 4;
      index = ((spin+1) >> 1) + (n1+n2+n3+n4) + 4;

      // The store back to global memory, not the branch or the RNG generation
      // seems to be the killer here.
      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept
          loc_grid[my_idx] = -1*spin;
      } 
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

  }

  return;

}