#include "../include/helpers.h"

cudaError_t err;  // cudaError_t is a type defined in cuda.h

// Boilerplate error checking code borrowed from stackoverflow
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

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
  
    cudaMemcpy(d_Pacc, h_Pacc, config->prob_size*sizeof(float), cudaMemcpyHostToDevice );
    free(h_Pacc);

  }

void preComputeNeighbours(ising_model_config *config, int *d_neighbour_list) {
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

  cudaMemcpy(d_neighbour_list, h_neighbour_list, neighbour_list_size*sizeof(int), cudaMemcpyHostToDevice );

  free(h_neighbour_list);

  /// Also store a version in constant memory
    //   uint8_t *hc_next = (uint8_t *)malloc(MAXL*sizeof(uint8_t));
    //   uint8_t *hc_prev = (uint8_t *)malloc(MAXL*sizeof(uint8_t));

    //   for (spin_index=0;spin_index<L;spin_index++){

    //     hc_next[spin_index] = (spin_index+1)%L;
    //     hc_prev[spin_index] = (spin_index+L-1)%L;

    //   }
}

// TODO: Deprecate this with the binary dump from issue #8
void outputGridToFile(ising_model_config *launch_struct, int *host_grid, float *host_mag, int iteration, int stream_ix) {
  /* Output the grid to a file.
      *
      * Parameters:
      * launch_struct: pointer to the ising model configuration
      * host_array: pointer to the array of spins
      * i: iteration number
  */
  fprintf(stderr, "Outputting grid to file\n");
  fflush(stderr);
  // Output the grid to a file
  char filename[100];
  int grid_size = launch_struct->size[0]*launch_struct->size[1];

  snprintf(filename, sizeof(filename), prefix);
  snprintf(filename+strlen(prefix), sizeof(filename)-strlen(prefix), "grid_%d_%d_%d.txt", stream_ix, grid_size, iteration);

  realpath(filename, filename);
  fprintf(stderr, "Making File: %s\n", filename);
  fflush(stderr);

  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }
  fprintf(stderr, "File Opened\n");
  fflush(stderr);

  fprintf(fp, "Grid Dims %d %d\n", launch_struct->size[0], launch_struct->size[1]);
  fprintf(stderr, "Grid Dims %d %d\n", launch_struct->size[0], launch_struct->size[1]);
  fflush(stderr);

  int row, col, grid_index;
  for (grid_index=0; grid_index<launch_struct->num_concurrent;grid_index++){
    fprintf(stderr, "Copy %d, Mag %f, Nucleated %d\n", grid_index+1, host_mag[grid_index], host_mag[grid_index] > launch_struct->nucleation_threshold);
    fprintf(fp, "Copy %d, Mag %f, Nucleated %d\n", grid_index+1, host_mag[grid_index], host_mag[grid_index] > launch_struct->nucleation_threshold);
    for (row=0;row<launch_struct->size[0];row++){
      for (col=0;col<launch_struct->size[1];col++){
        //fprintf(stderr, "%d ", host_grid[row*launch_struct->size[0]+col+grid_index*grid_size]);
        fprintf(fp, "%d ", host_grid[row*launch_struct->size[0]+col+grid_index*grid_size]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

}
