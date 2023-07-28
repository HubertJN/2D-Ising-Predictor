#include <iostream>
#include <fstream>

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
void FoutputGridToFile(ising_model_config *launch_struct, int *host_grid, float *host_mag, int iteration, int stream_ix) {
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
  snprintf(filename+strlen(prefix), sizeof(filename)-strlen(prefix), "grid_%d_%d_%d.dat", stream_ix, grid_size, iteration);

  realpath(filename, filename);
  fprintf(stderr, "Making File: %s\n", filename);
  fflush(stderr);

  std::fstream file;
  bool write_err = false;
  file.open(filename, std::ios::out|std::ios::binary);

  const size_t size_sz = sizeof(size_t); //Use size_t type for header data
  const size_t host_mag_sz = sizeof(host_mag[0]);
  const size_t host_grid_sz = sizeof(host_grid[0]);
  fprintf(stderr, "Size of size_t: %d\n", size_sz);

  size_t next_location = (size_t) file.tellg() + size_sz*2 + host_mag_sz; // + GIT_VERSION_SIZE*sizeof(char);
  
  //Write Size of ints and data, IO verification constant
  const float io_verify = 3.0/32.0;// An identifiable value - use to check type, endianness etc TODO - match to host_mag type
  file.write((char*) & host_mag_sz, size_sz); // Size of mag data
  file.write((char*) & host_grid_sz, size_sz); // Size of grid data
  file.write((char*) &io_verify, host_mag_sz);
 
  //Check file location matches what we expected
  if((size_t)file.tellg() != next_location) write_err=1;
  fprintf(stderr, "Next loc %d (%d)\n", next_location, write_err);

  //Now write dimension info: n_dims, followed by each dim, and the total number of grids
  const size_t n_dims = 2; // TODO - n_dims should be got from somewhere higher up!
  size_t tmp;
  next_location += size_sz*(n_dims+3); // + 3 for location info, n_dims, n_grids
  file.write((char*) & next_location, size_sz);
  file.write((char*) & n_dims, size_sz);
  tmp = launch_struct->size[0]; // todo fix to use n_dims properly
  file.write((char*) & tmp, size_sz);
  tmp = launch_struct->size[1];
  file.write((char*) & tmp, size_sz);
  tmp = launch_struct->num_concurrent;
  file.write((char*) & tmp, size_sz);

  fprintf(stderr, "n_dims %d, dims % d %d, n_conc %d\n", n_dims,  launch_struct->size[0],  launch_struct->size[1], launch_struct->num_concurrent);
  //Check file location matches what we expected
  if((size_t)file.tellg() != next_location) write_err=1;
  fprintf(stderr, "Next loc %d (%d)\n", next_location, write_err);
  // todo remove these debugging prints

  const size_t total_size = launch_struct->size[0] * launch_struct->size[1];

  // First write info on all the grids - index, magnetisation, nucleation state
  const size_t grid_info_count = (size_sz*2 + host_mag_sz) * launch_struct->num_concurrent;
  fprintf(stderr, "grid info count %d\n", grid_info_count);
  next_location += grid_info_count + size_sz;
  file.write((char*) & next_location, size_sz);


  size_t grid_index;
  fprintf(stderr, "mag sz %d, grid sz %d\n", host_mag_sz, host_grid_sz);
  for (grid_index=0; grid_index<launch_struct->num_concurrent;grid_index++){
    file.write((char*) & grid_index, size_sz);
    file.write((char*) & host_mag[grid_index], host_mag_sz);
    size_t nuc = host_mag[grid_index] > launch_struct->nucleation_threshold;
    file.write((char*) &nuc, size_sz);

   // Also print this info to the screen
   fprintf(stderr, "Copy %d, Mag %f, Nucleated %d\n", grid_index+1, host_mag[grid_index], nuc);
  }

  //Check file location matches what we expected
  if((size_t)file.tellg() != next_location) write_err=1;
  fprintf(stderr, "Next loc %d (%d)\n", next_location, write_err);

  // Then write the actual grids

  for(grid_index=0; grid_index<launch_struct->num_concurrent;grid_index++){
    next_location += host_grid_sz*total_size + size_sz;
    file.write((char*) & next_location, size_sz);
    file.write((char*) (host_grid + grid_index*total_size), total_size*host_grid_sz);
  }

  //Check file location matches what we expected
  if((size_t)file.tellg() != next_location) write_err=1;
  fprintf(stderr, "Next loc %d\n", next_location);

  if(write_err) fprintf(stderr, "File Writing error");

  file.close();
}

int readGridsFromFile(ising_model_config * config, int * &host_grid, char* filename){
  /* Read grids from given file
      *
      * Parameters:
      * config : pointer to config struct (allocated)
      * host_grid : pointer to grid (will be allocated/reallocated)
      * filename : full filename to read from
    Returns 1 if there is an error
   */

  fprintf(stderr, "Reading grid from file\n");
  fflush(stderr);

  std::fstream file;
  bool read_err = false;
  file.open(filename, std::ios::in|std::ios::binary);

  // Read file header info
  const size_t size_sz = sizeof(size_t); //Use size_t type for header data - todo - add to file and read back...
  size_t host_mag_sz;
  size_t host_grid_sz;
  size_t next_location;

  //Read Size of ints and data, IO verification constant
  const float io_verify = 3.0/32.0;// An identifiable value - use to check type, endianness etc TODO - match to host_mag type
  float io_verify_in;
  file.read((char*) & host_mag_sz, size_sz); // Size of mag data
  file.read((char*) & host_grid_sz, size_sz); // Size of grid data
  file.read((char*) &io_verify_in, host_mag_sz);

  fprintf(stderr, "Host mag size %d\n", host_mag_sz);
  fprintf(stderr, "Host grid sz %d\n", host_grid_sz);
  fprintf(stderr, "IO verify %f\n", io_verify_in);
  // Do verification steps:
  if(host_grid_sz != sizeof(host_grid[0])){
    fprintf(stderr, "File data does not match requested data size, aborting");
    // todo - avoid this potential error somehow?
    return 1;
  }
  if(io_verify_in != io_verify){
    fprintf(stderr, "IO verification failed, aborting");
    // todo - is exact real comparision the right thing here?
    return 1;
  }

  // Read grids metadata
  // todo Do we want to fill in the config struct? Or verify against it??

  // Next location marker
  file.read((char*) & next_location, size_sz);
  
  size_t n_dims;
  file.read((char*) & n_dims, size_sz);

  size_t tmp1, tmp2; // todo fix for different n_dims?
  file.read((char*) & tmp1, size_sz);
  file.read((char*) & tmp2, size_sz);

  size_t n_c;
  file.read((char*) & n_c, size_sz);

  // todo - see above
  // For now, verify against config passed in...??
  if(n_dims != 2 || tmp1 != config->size[0] || tmp2 != config->size[1] || n_c != config->num_concurrent){
    fprintf(stderr, "Data read does not match specified configuration");
    return 2;
  }

  const size_t total_size = config->size[0] * config->size[1];

  host_grid = (int *) malloc(n_c * total_size * sizeof(int));
  //Skip over magnetization and nucleation - should calculate instead
  file.read((char*) & next_location, size_sz);
  file.seekg(next_location);

  // Read grids
  size_t grid_index;
  for (grid_index=0; grid_index<config->num_concurrent; grid_index++){

    file.read((char*) & next_location, size_sz);
    file.read((char*) (host_grid + grid_index*total_size), total_size*host_grid_sz);
  }

  // Finish
  file.close();
  return 0;

}



