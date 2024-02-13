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
  for (grid_index=0; grid_index<config->size[0]*config->size[1]; grid_index++){

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
void outputGridToTxtFile(ising_model_config *launch_struct, int *host_grid, float *host_mag, int iteration, int stream_ix) {
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
  char pathname[100];
  int grid_size = launch_struct->size[0]*launch_struct->size[1];

  namespace fs = std::filesystem;
  fs::path pathname_path;

  pathname_path = fs::current_path();
  path_search(pathname_path, pathname);

  snprintf(filename, sizeof(filename), prefix);
  snprintf(filename+strlen(prefix), sizeof(filename)-strlen(prefix), "grid_%d_%d_%d.dat", stream_ix, grid_size, iteration);

  snprintf(pathname+strlen(pathname), sizeof(pathname)-strlen(filename), "/%s", filename);
  strcpy(filename, pathname);
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

// function that finds path to GPU-Arch-Test provided gasp is run from within repo
int path_search(std::filesystem::path pathname_path, char* pathname) {
  if (pathname_path.filename() == "2DIsing_Model") {
    std::string pathname_str = pathname_path.string();
    strcpy(pathname, pathname_str.c_str());
    fprintf(stderr, "%s\n", pathname);
    return 0;
  }
  pathname_path = pathname_path.parent_path();
  path_search(pathname_path, pathname);
  return 1;
}

char* getFileUuid(){

  char lett[27]="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  char * str;
  str = (char*) malloc((FILE_UUID_LEN+1) * sizeof(char));
  for (int i = 0; i < FILE_UUID_LEN; i++){
    str[i] = lett[rand() % 26];
  }
  str[FILE_UUID_LEN]='\0';
  return str;
}

void outputModelId(std::fstream & file, file_handle& gridHdl, int i_conc, ising_model_config * launch_struct){

  char info[100];
  snprintf(info, 100, "Model number: %d with id %s \n", i_conc, gridHdl.uuid);
  fprintf(stderr, info);
  file.write(info, strlen(info));
}

void fillCompletePath(char* filename){
  // Get current working directory
  fprintf(stderr, "Step 0 %s\n", filename); 
 
   char cwd[PATH_MAX];
   if (getcwd(cwd, sizeof(cwd)) != NULL) {
       printf("Current working dir: %s\n", cwd);
   } else {
       perror("getcwd() error");
   }
  // compare to expected path from git repo
  // test last 3 chars are "bin"
  if (strcmp(cwd+strlen(cwd)-3, "bin") == 0) {
    // filepath is ../prefix
    snprintf(filename, PATH_MAX, "../%s", prefix);
  }
  // test last 11 chars are "2DIsing_Model"
  else if (strcmp(cwd+strlen(cwd)-11, "2DIsing_Model") == 0) {
    snprintf(filename, PATH_MAX, "./%s", prefix);
  }
  // test if last chars are "GASP"
  else if (strcmp(cwd+strlen(cwd)-4, "GASP") == 0) {
    snprintf(filename, PATH_MAX, "./%s", prefix); 
  }
  // exit if not running from within repo
  else {
    fprintf(stderr, "Error: not running from any expected location, please launch from bin or main repo\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Step 1 %s\n", filename); 
  
  // resolve the path using realpath
  char buffer[PATH_MAX];
  char *res = realpath(filename, buffer);
  if (res) {
    snprintf(filename, PATH_MAX, buffer);
  } else {
    char* errStr = strerror(errno);
    fprintf(stderr, "filepath error: %s\n", errStr);
    perror("realpath");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Step 2 %s\n", filename); 
 
}

// \todo Swap mag for committor, remove nucleation info
// \todo Add extra check info
void outputInitialInfo(file_handle &theHdl, ising_model_config *launch_struct, int stream_ix, int i_conc) {
 
//void outputGridToFile(ising_model_config *launch_struct, int *host_grid, float *host_mag, int iteration, int stream_ix) {
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
  char filename[PATH_MAX];
  int grid_size = launch_struct->size[0]*launch_struct->size[1];

  fillCompletePath(filename);
  fprintf(stderr, "Step 3 %s\n", filename); 
 
  char * uid = getFileUuid();
  snprintf(theHdl.uuid, FILE_UUID_LEN+1, uid);

  snprintf(filename+strlen(filename), sizeof(filename)-strlen(filename), "/grid_%s_%d_%d.dat", uid, stream_ix, grid_size);
  fprintf(stderr, "Making File: %s\n", filename);
  fflush(stderr);

  snprintf(theHdl.filename, sizeof(filename), filename); // Stash filename for later

  bool write_err = false;
  theHdl.file.open(filename, std::ios::out|std::ios::binary);

  theHdl.size_sz = sizeof(size_t); //Use size_t type for header data
  theHdl.host_grid_sz = sizeof(int);

  const size_t VERSION_SIZE = 10+1; // Add 1 Char for null term
  char version_str[VERSION_SIZE];
  snprintf(version_str, VERSION_SIZE, VERSION);
  //fprintf(stderr, "Version %s\n", version_str);
  theHdl.next_location = (size_t) theHdl.file.tellg() + theHdl.size_sz + sizeof(float) + VERSION_SIZE*sizeof(char);

  //Write Size of ints and data, IO verification constant
  const float io_verify = 3.0/32.0;// An identifiable value - use to check type, endianness etc. Note this should probably match host_mag type
  theHdl.file.write((char*) & theHdl.host_grid_sz, theHdl.size_sz); // Size of grid data
  theHdl.file.write((char*) &io_verify, sizeof(float));
  theHdl.file.write( version_str, VERSION_SIZE); // Version string info

  //Check file location matches what we expected
  if((size_t)theHdl.file.tellg() != theHdl.next_location) write_err=1;

  //Now write dimension info: n_dims, followed by each dim, and the total number of grids
  size_t tmp;
  theHdl.next_location += theHdl.size_sz*(n_dims+3); // + 3 for location info, n_dims, n_grids
  theHdl.file.write((char*) & theHdl.next_location, theHdl.size_sz);
  theHdl.file.write((char*) & n_dims, theHdl.size_sz);
  for(int i = 0; i< n_dims; i++){
    tmp = launch_struct->size[i];
    theHdl.file.write((char*) & tmp, theHdl.size_sz);
  }
  tmp = launch_struct->num_concurrent;
  theHdl.file.write((char*) & tmp, theHdl.size_sz);

  //Check file location matches what we expected
  if((size_t)theHdl.file.tellg() != theHdl.next_location) write_err=1;
 
  // Stash els per grid
  theHdl.grid_els = grid_size;

  if(!write_err){
    fprintf(stderr, "File opened and metadata written successfully\n");
  }else{
    fprintf(stderr, "Error opening file or writing metadata\n");
  }

}

void writeSingleGrid(file_handle &theHdl, int *host_grid, int iteration, int stream_ix) {

  // Write a single grid
  bool write_err = false;

  theHdl.next_location += theHdl.host_grid_sz* theHdl.grid_els + theHdl.size_sz;
  theHdl.file.write((char*) & theHdl.next_location, theHdl.size_sz);
  theHdl.file.write((char*) host_grid, theHdl.grid_els*theHdl.host_grid_sz);

  //Check file location matches what we expected
  if((size_t)theHdl.file.tellg() != theHdl.next_location) write_err=1;

  if(write_err) fprintf(stderr, "File Writing error\n");

}

void finaliseFile(file_handle &theHdl){

  theHdl.file.close();
}

//\todo Update to match writer
int readGridsFromFile(ising_model_config * config, int * &host_grid){
  /* Read grids from given file
      *
      * Parameters:
      * config : pointer to config struct (allocated)
      * host_grid : pointer to grid (will be allocated/reallocated)
      * filename : full filename to read from
    Returns 1 if there is an error
   */

  fprintf(stderr, "Reading grid from file: %s\n", config->input_file);
  fflush(stderr);

  std::fstream file;
  file.open(config->input_file, std::ios::in|std::ios::binary);

  // Read file header info
  const size_t size_sz = sizeof(size_t); //Use size_t type for header data - todo - add to file and read back...
  size_t host_mag_sz;
  size_t host_grid_sz;
  size_t next_location;

  //Read Size of ints and data, IO verification constant
  const float io_verify = 3.0/32.0;// An identifiable value - use to check type, endianness etc
  float io_verify_in;
  file.read((char*) & host_mag_sz, size_sz); // Size of mag data
  file.read((char*) & host_grid_sz, size_sz); // Size of grid data
  file.read((char*) &io_verify_in, host_mag_sz);

  const size_t VERSION_SIZE = 10+1; // Add 1 Char for null term
  char version_str[VERSION_SIZE];
  file.read(version_str, VERSION_SIZE);

  fprintf(stderr, "File generated by code version %s\n", version_str);

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
  
  size_t n_dims_in;
  file.read((char*) & n_dims_in, size_sz);

  size_t tmp1, tmp2; // todo fix for different n_dims?
  file.read((char*) & tmp1, size_sz);
  file.read((char*) & tmp2, size_sz);

  size_t n_c;
  file.read((char*) & n_c, size_sz);

  // todo - see above
  // For now, verify against config passed in...??
  if(n_dims_in != n_dims || tmp1 != config->size[0] || tmp2 != config->size[1] || n_c != config->num_concurrent){
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



