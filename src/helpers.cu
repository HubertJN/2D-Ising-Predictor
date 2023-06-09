#include "../include/helpers.h"

// populate acceptance probabilities
void preComputeProbs_gpu(ising_model_config *config, float* d_Pacc) {
  /* Precompute the acceptance probabilities for the GPU.
    *
    * Parameters:
    * beta: inverse temperature
    * h: magnetic field
  */

    float *h_Pacc=(float *)malloc(config.prob_size*sizeof(float));

    int s, nsum, index;  
    for (s=-1;s<2;s=s+2){
      for (nsum=-4;nsum<5;nsum=nsum+2){
        index = 5*(s+1) + nsum + 4;
        h_Pacc[index] = expf(-(float)beta*2.0f*(float)s*((float)nsum+(float)h));
      }
    }
  
    gpuErrchk( cudaMemcpyToSymbol(d_Pacc, h_Pacc, config.prob_size*sizeof(float),0, cudaMemcpyHostToDevice ) );
    free(h_Pacc);

  }

void preComputeNeighbours(ising_model_config *config, int *d_neighbour_list){
  /* Precompute the neighbour list for the GPU.
      *
      * Parameters:
      * size: dimensions of the grid, 2d array
      * d_ising_grids: pointer to the array of grids
      * d_neighbour_list: pointer to the array of neighbour lists
  */

  // These could probably be cached in shared memory since they are the same for all threads.

  int *h_neighbour_list = (int *)malloc(config->size[0]*config->size[1]*4*sizeof(int));

  int spin_index;
  for (spin_index=0;spin_index<L*L;spin_index++){

    int row = spin_index/L;
    int col = spin_index%L;

    h_neighbour_list[4*(row*L+col) + 0] = L*((row+1)%L) + col;
    h_neighbour_list[4*(row*L+col) + 1] = L*((row+L-1)%L) + col;
    h_neighbour_list[4*(row*L+col) + 2] = L*row + (col+1)%L;
    h_neighbour_list[4*(row*L+col) + 3] = L*row + (col+L-1)%L;

  }

  gpuErrchk( cudaMemcpyToSymbol(d_neighbour_list, h_neighbour_list, 4*L*L*sizeof(int), cudaMemcpyHostToDevice ) );

  free(h_neighbour_list); 

  /// Also store a version in constant memory
    //   uint8_t *hc_next = (uint8_t *)malloc(MAXL*sizeof(uint8_t));
    //   uint8_t *hc_prev = (uint8_t *)malloc(MAXL*sizeof(uint8_t));

    //   for (spin_index=0;spin_index<L;spin_index++){

    //     hc_next[spin_index] = (spin_index+1)%L;
    //     hc_prev[spin_index] = (spin_index+L-1)%L;

    //   }
}