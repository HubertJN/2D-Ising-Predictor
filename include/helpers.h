#ifndef HELPERS_H
#define HELPERS_H
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "shared_data.h"
#include "input_reader.h"

// Macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Prototype for gpuAssert
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

void preComputeProbs(ising_model_config *config, float* d_Pacc);

void preComputeNeighbours(ising_model_config *config, int *d_neighbour_list);

// This needs to be set to whatever the containing folder is called
const char project_name[30]="2DIsing_Model";
// this is fine
const char prefix[30]="grid_binaries/output/";

typedef struct file_hdle{
  std::fstream file;
  char filename[PATH_MAX];
  char uuid[FILE_UUID_LEN+1];
  size_t next_location; // Allows to check file ptr has not moved since we last saw it
  size_t size_sz;
  size_t host_grid_sz;
  size_t grid_els; 
  size_t num_grids;
} file_handle;

void outputGridToTxtFile(ising_model_config *config, int *host_grid, float *host_mag, int iteration, int stream_ix);

char* getFileUuid();
void outputModelId(std::fstream & file, file_handle& gridHdl, ising_model_config * launch_struct);
void outputInitialInfo(file_handle &theHdl, ising_model_config *launch_struct, int stream_ix);
void writeSingleGrid(file_handle &theHdl, int *host_grid, int iteration, int stream_ix);
void writeAllGrids(file_handle &theHdl, int *host_grid, int iteration, int stream_ix);
void finaliseFile(file_handle &theHdl);



void outputGridToFile(ising_model_config *config, int *host_grid, float *host_mag, int iteration, int stream_ix);

int readGridsFromFile(ising_model_config * config, int *&host_grid);

int path_search(std::filesystem::path pathname_path, char* pathname);
void fillCompletePath(char* filename);
 
#endif // HELPERS_H

