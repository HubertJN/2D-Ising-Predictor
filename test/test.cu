#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "../include/input_reader.h"

int main( int argc, char *argv[]) {
    char* filename = "input.txt";
    // Check if command line arguments were provided
    if (argc > 1) {
        filename = argv[1];
    }
    // pattern for further arguments
    // if (argc > 2) {
    //     count = atoi(argv[2]);
    // } 
    if (argc > 2) {
        printf("Usage: %s [filename] \n", argv[0]);
        exit(1);
    }
    int models;
    get_number_of_models(filename, &models);
    // Allocate memory for the array of pointers to structs
    ising_model_config* params_array[models];
    if (params_array == NULL) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        exit(1);
    }
    read_input_file(filename, params_array, models);
    fprintf(stderr, "Number of models: %d\n", sizeof(params_array)/sizeof(params_array[0]));
    
    // Rest of the program...
    
    return 0;
    }