#ifndef INPUT_READER_H
#define INPUT_READER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/kvp.h"

// structure to hold the configuration of the ising model, not all parameters are used in all models
typedef struct ising_model_config {
    // User set parameters in config file
    int model_id;     // model id
    int num_concurrent; // number of concurrent simulations
    int num_threads; // number of threads per simulation
    int size[2];         // size of each grid, 2D
    int iterations;   // number of iterations in the simulation
    int iter_per_step; // number of iterations per step
    int seed;         // seed for the random number generator
    float temperature;  // temperature of the system
    // System set parameters
    int num_threads;  // number of threads per block
    int num_blocks;   // number of blocks
} ising_model_config;



void read_lines(FILE* input_file, int start_line, int end_line);

void get_number_of_models(const char* filename, int* models);

void read_input_file(const char* filename, ising_model_config* params_array[], int models);



#endif // INPUT_READER_H