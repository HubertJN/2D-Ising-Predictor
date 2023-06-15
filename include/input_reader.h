#ifndef INPUT_READER_H
#define INPUT_READER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/kvp.h"

// structure to hold the configuration of the ising model, not all parameters are used in all models
// NOTE: when adding new user set parameters update the read_input_file function in input_reader.cu
typedef struct ising_model_config {
    // User set parameters in config file
    int model_id;     // model id
    int num_concurrent; // number of concurrent simulations
    int size[2];         // size of each grid, 2D
    int iterations;   // number of iterations in the simulation
    int iter_per_step; // number of iterations per step
    int seed;         // seed for the random number generator
    float inv_temperature;  // temperature of the system
    float field;        // magnetic field strength
    char* input_file; // input file name
    int starting_config; // starting configuration, 0 for file input, 1 for random, 2 for all up, 3 for all down
    // User or System set parameters
    int num_threads;  // number of threads per block
    // System set parameters
    int num_blocks;   // number of blocks
    int element_size; // size of the grid elements in bytes
    int mem_size;     // size of all grids required in bytes
    int prob_size;    // size of the probability array
} ising_model_config;



void read_lines(FILE* input_file, int start_line, int end_line);

void get_number_of_models(const char* filename, int* models);

void read_input_file(const char* filename, ising_model_config* params_array[], int models);

void load_grid(cudaStream_t stream, ising_model_config* launch_struct, int* host_grid, int* dev_grid);


#endif // INPUT_READER_H