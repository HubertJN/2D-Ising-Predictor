#include <stdio.h>
#include <stdlib.h>

#include "input_reader.h"

void main( int argc, char *argv[]) {
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
    ising_model_config** params_array = read_input_file(filename);
    printf("Number of models: %d", sizeof(params_array));
    
    // Rest of the program...
    
    }