
#include "../include/input_reader.h"


void register_inputs(ising_model_config* params) {
    // For each new model config update the kvp pointers
    kvp_register_i("model_id", &params -> model_id);
    kvp_register_i("num_concurrent", &params -> num_concurrent);
    kvp_register_i("size_x", &params -> size[0]);
    kvp_register_i("size_y", &params -> size[1]);
    kvp_register_i("iterations", &params -> iterations);
    kvp_register_i("iter_per_step", &params -> iter_per_step);
    kvp_register_i("seed", &params -> seed);
    kvp_register_f("temperature", &params -> temperature);
}

void read_lines(FILE* input_file, int start_line, int end_line) {
    char* line;
    size_t len = 0;
    int line_num = 0;
    while ((getline(&line, &len, input_file)) != -1) {
        if (line_num >= start_line && line_num < end_line) {
            kvp_from_text(line);
        }
        line_num++;
    }
}

void get_number_of_models(const char* filename, int* models) {
   FILE* input_file = fopen(filename, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        exit(1);
    } 

    char* line;
    size_t len = 0;
    while ((getline(&line, &len, input_file)) != -1) {
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
        params_array[i] = (ising_model_config*)malloc(sizeof(ising_model_config));
        if (params_array[i] == NULL) {
            fprintf(stderr, "Error: Could not allocate memory\n");
            exit(1);
        }
    }

    char* line;
    size_t len = 0;

    // Get the linenumbers for each new model
    int line_numbers[models];
    int model_num = 0;
    while ((getline(&line, &len, input_file)) != -1) {
        if (strcmp(line, "## New model ##\n") == 0) {
            line_numbers[model_num] = ftell(input_file);
            model_num++;
        }
    }

    
    ising_model_config* params;
    // Modify this loop to go over the line numbers instead of parsing the file line by line
    for (int i = 0; i < models; i++) {
        params = params_array[models];
        // ReRegister the inputs to update the pointers
        register_inputs(params);
        
        // get the line number pairs
        if (i == models - 1) {
            // read from line_numbers[i] to eof
            read_lines(input_file, line_numbers[i], -1);
        }
        else {
            // read from line_numbers[i] to line_numbers[i+1]
            read_lines(input_file, line_numbers[i], line_numbers[i+1]);
        }
        
    }

    // Close the file and free the memory
    fclose(input_file);
}
