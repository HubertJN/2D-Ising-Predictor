#include "../include/helpers.h"

// populate acceptance probabilities
void preComputeProbs(ising_model_config *config, float* d_Pacc) {
  /* Precompute the acceptance probabilities for the GPU.
    *
    * Parameters:
    * beta: inverse temperature
    * h: magnetic field
  */

    float *h_Pacc=(float *)malloc(config->prob_size*sizeof(float));
    
    int s, nsum, index;  
    for (s=-1;s<2;s=s+2){
      for (nsum=-4;nsum<5;nsum=nsum+2){
        index = ((s+1) >> 1) + (nsum) + 4;
        h_Pacc[index] = expf(-(float)config->inv_temperature*2.0f*(float)s*((float)nsum+(float)config->field));
      }
    }
  
    cudaMemcpyToSymbol(d_Pacc, h_Pacc, config->prob_size*sizeof(float),0, cudaMemcpyHostToDevice );
    free(h_Pacc);

  }

void preComputeNeighbours(ising_model_config *config, int *d_neighbour_list){
  /* Precompute the neighbour list for the GPU.
      *
      * Parameters:
      * ising_model_config: pointer to the ising model configuration
      * d_neighbour_list: pointer to the array of neighbour lists
  */

  // These could probably be cached in shared memory since they are the same for all threads.

  // Allocate memory for the neighbour list this is 4x the size of the lattice
  int neighbour_list_size = 4*config->size[0]*config->size[1];
  int *h_neighbour_list = (int *)malloc(neighbour_list_size*sizeof(int));

  int grid_index;
  for (grid_index=0;grid_index<config->size[0]*config->size[1];grid_index++){

    int row = grid_index/config->size[0];
    int col = grid_index%config->size[0];

    h_neighbour_list[4*(row*config->size[0]+col) + 0] = config->size[0]*((row+1)%config->size[0]) + col;
    h_neighbour_list[4*(row*config->size[0]+col) + 1] = config->size[0]*((row+config->size[0]-1)%config->size[0]) + col;
    h_neighbour_list[4*(row*config->size[0]+col) + 2] = config->size[0]*row + (col+1)%config->size[0];
    h_neighbour_list[4*(row*config->size[0]+col) + 3] = config->size[0]*row + (col+config->size[0]-1)%config->size[0];

  }

  cudaMemcpyToSymbol(d_neighbour_list, h_neighbour_list, neighbour_list_size*sizeof(int), cudaMemcpyHostToDevice );

  free(h_neighbour_list);

  /// Also store a version in constant memory
    //   uint8_t *hc_next = (uint8_t *)malloc(MAXL*sizeof(uint8_t));
    //   uint8_t *hc_prev = (uint8_t *)malloc(MAXL*sizeof(uint8_t));

    //   for (spin_index=0;spin_index<L;spin_index++){

    //     hc_next[spin_index] = (spin_index+1)%L;
    //     hc_prev[spin_index] = (spin_index+L-1)%L;

    //   }
}