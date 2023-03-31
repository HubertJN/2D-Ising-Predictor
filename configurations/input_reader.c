#include <stdio.h>
#include <stdlib.h>
#include "input_reader.h"

ising_model_config** read_input_file(const char* filename, int* num_params) {
    FILE* input_file = fopen(filename, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        exit(1);
    }

    // Count the number of models defined in the file
    int models = 0;
    
    char* line;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&line, &len, input_file)) != -1) {
        if (strcmp(line, "## New model ##\n") == 0) {
            models++;
        }
    }
    fseek(input_file, 0, SEEK_SET);  // Reset file pointer to beginning of file

    // Allocate memory for the array of pointers to structs
    ising_model_config** params_array = malloc(sizeof(ising_model_config*) * models);
    if (params_array == NULL) {
        fprintf(stderr, "Error: Could not allocate memory\n");
        exit(1);
    }

    // Read the file line by line and fill the array of pointers to structs
    int model_num = 0;
    char *ptr;
    char *ret_str;
    long *ret_int;

    while ((read = getline(&line, &len, input_file)) != -1) {
        // Check line for new model keyword
        if (strcmp(line, "## New model ##\n") == 0) {
            // Allocate memory for the struct and add it to the array
            ising_model_config* params = malloc(sizeof(ising_model_config));
            params_array[model_num] = params;
            if (params == NULL) {
                fprintf(stderr, "Error: Could not allocate memory\n");
                exit(1);
            }
            // Increment the model number
            model_num++;
        } else if (strcmp(line, "\n") == 0) {
            // If an empty line was found, do nothing
        } else {
            // If a new model or empty line was not found, the line must be a parameter
            // Read the line and fill the struct
            ret_str = strchr(line, '=');
            // Convert the string following the = to an int
            ret_int = strtol(ret_str, &ptr, 10);
            
            // test the line for a parameter keyword if found fill the struct
            if (strcmp(line, "size_x") == 0) {
                params -> size[0] = ret;
            }
            else if (strcmp(line, "size_y") == 0) {
                params -> size[1] = ret;
            }
            else if (strcmp(ptr, "temperature") == 0) {
                params -> temperature = ret;
            }
            else if (strcmp(ptr, "iterations") == 0) {
                params -> iterations = ret;
            }
            else if (strcmp(ptr, "iter_per_step") == 0) {
                params -> iter_per_step = ret;
            }
            else if (strcmp(ptr, "num_concurrent") == 0) {
                params -> num_concurrent = ret;
            }
            else {
                fprintf(stderr, "Error: Unknown parameter '%s'\n", line);
                exit(1);
            }
        }
    }

    // Close the file and free the memory
    fclose(input_file);
    free(line);
    free(ptr);
    free(ret_str);
    free(ret_int);
    free(read);
    free(model_num);
    free(len);

    // Return the array of pointers to structs
    return params_array;
}
