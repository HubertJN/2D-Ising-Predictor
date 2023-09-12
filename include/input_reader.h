#ifndef INPUT_READER_H
#define INPUT_READER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared_data.h"
#include "../include/kvp.h"
#include "../include/helpers.h"


void read_lines(FILE* input_file, int start_line, int end_line);

void get_number_of_models(const char* filename, int* models);

void read_input_file(const char* filename, ising_model_config* params_array[], int models);

void load_grid(cudaStream_t stream, ising_model_config* launch_struct, int* host_grid, int* dev_grid);


#endif // INPUT_READER_H
