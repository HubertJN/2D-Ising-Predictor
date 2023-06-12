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


// sweep on the gpu - default version
__global__ void mc_sweep(curandState *state, const int L, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {
  /* 
    * Default version of the sweep kernel, uses a neighbour list to avoid branching.
    *
    * 
    * Parameters:
    * state: pointer to the RNG state array
    * L: linear size of the grid
    * ngrids: number of grids
    * d_ising_grids: pointer to the array of grids
    * d_neighbour_list: pointer to the array of neighbour lists
    * nsweeps: number of sweeps to perform
  */

  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int index;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    int N = L*L;
    // Avoid rounding errors after creating random numbers by ensuring out max
    // is upto 1.0f not up to 1.0f + FLT_EPSILON
    float shrink = (1.0f - FLT_EPSILON)*(float)N;

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*N]; // pointer to device global memory 


    int imove, my_idx, spin, n1, n2, n3, n4, row, col;
    for (imove=0;imove<N*nsweeps;imove++){

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));

      row = my_idx/L;
      col = my_idx%L;

      spin = loc_grid[my_idx];

      // find neighbours, periodic boundary conditions. D,U,L,R
      n1 = loc_grid[L*((row+1)%L) + col];
      n2 = loc_grid[L*((row+L-1)%L) + col];
      n3 = loc_grid[L*row + (col+1)%L];
      n4 = loc_grid[L*row + (col+L-1)%L];

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