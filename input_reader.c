#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "input_reader.h"


void get_number_of_models(const char* filename, int* models) {
   FILE* input_file = fopen(filename, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        exit(1);
    } 

    char* line;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&line, &len, input_file)) != -1) {
        if (strcmp(line, "## New model ##\n") == 0) {
            *models = *models + 1;
        }
    }
    // Close the file and free the memory
    fclose(input_file);
}

void read_input_file(const char* filename, ising_model_config* params_array[], int models) {
    FILE* input_file = fopen(filename, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        exit(1);
    }

    // Creater a struct for each model
    for (int i = 0; i < models; i++) {
        params_array[i] = malloc(sizeof(ising_model_config));
        if (params_array[i] == NULL) {
            fprintf(stderr, "Error: Could not allocate memory\n");
            exit(1);
        }
    }

    // Read the file line by line and fill the array of pointers to structs
    char* line;
    size_t len = 0;
    ssize_t read;
    int model_num = 0;
    char *ptr;
    char *ret_str;
    long ret_int;

    ising_model_config* params;

    int *result;
    while ((read = getline(&line, &len, input_file)) != -1) {
        // Check line for new model keyword
        if (strcmp(line, "## New model ##\n") == 0) {
            // use the model_num to get the pointer to the struct
            params = params_array[model_num];
            // Increment the model number
            model_num++;
        } else if (strcmp(line, "\n") == 0) {
            // If an empty line was found, do nothing
            continue;
        } else {
            // If a new model or empty line was not found, the line must be a parameter
            // Read the line and fill the struct, 
            // get the pointer to '='
            ret_str = strchr(line, '=');
            // Convert the string following the = to an int, increment the pointer to get the next char
            ret_int = strtol(ret_str+1, &ptr, 10);
            
            // test the line for a parameter keyword if found fill the struct
            if (strstr(line, "size_x") > 0) {
                params -> size[0] = ret_int;
            }
            else if (strstr(line, "size_y") > 0) {
                params -> size[1] = ret_int;
            }
            else if (strstr(line, "temperature") > 0) {
                params -> temperature = ret_int;
            }
            else if (strstr(line, "iterations") > 0) {
                params -> iterations = ret_int;
            }
            else if (strstr(line, "iter_per_step") > 0) {
                params -> iter_per_step = ret_int;
            }
            else if (strstr(line, "num_concurrent") > 0) {
                params -> num_concurrent = ret_int;
            }
            else {
                fprintf(stderr, "Error: Unknown parameter '%s', '%d'\n", line, ret_int);
                exit(1);
            }
        }
    }

    // Close the file and free the memory
    fclose(input_file);
    // free(line);
    // free(ptr);
    // free(ret_str);
    // free(ret_int);
    // free(read);
    // free(model_num);
    // free(len);
}
