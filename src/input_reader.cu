
#include "../include/input_reader.h"

void read_lines(FILE* input_file, int start_line, int end_line) {
    char* line;
    size_t len = 0;
    int line_num = 0;
    //reset the file to begining
    fseek(input_file, 0, SEEK_SET);
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

    // Create a struct for each model
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
    int line_num = 0;
    while ((getline(&line, &len, input_file)) != -1) {
        if (strcmp(line, "## New model ##\n") == 0) {
            fprintf(stderr, "Found new model at line %d\n", line_num);
            line_numbers[model_num] = line_num;
            model_num++;
        }
        line_num++;
    }
    // Preserve line num to use as end line later

    // throwaway variables for kvp, match the struct
    int model_id;     // model id
    int num_concurrent; // number of concurrent simulations
    int size[2];         // size of each grid, 2D
    int iterations;   // number of iterations in the simulation
    int iter_per_step; // number of iterations per step
    int seed;         // seed for the random number generator
    float temperature; 
    kvp_register_i("model_id", &model_id);
    kvp_register_i("num_concurrent", &num_concurrent);
    kvp_register_i("size_x", &size[0]);
    kvp_register_i("size_y", &size[1]);
    kvp_register_i("iterations", &iterations);
    kvp_register_i("iter_per_step", &iter_per_step);
    kvp_register_i("seed", &seed);
    kvp_register_f("temperature", &temperature);
    

    // Modify this loop to go over the line numbers instead of parsing the file line by line
    for (int i = 0; i < models; i++) {
        // get the line number pairs
        if (i == models - 1) {
            // read from line_numbers[i] to eof
            fprintf(stderr, "Reading from line %d to eof\n", line_numbers[i]);
            read_lines(input_file, line_numbers[i], line_num);
            printf("beep\n");
        }
        else {
            // read from line_numbers[i] to line_numbers[i+1]
            fprintf(stderr, "Reading from line %d to line %d\n", line_numbers[i], line_numbers[i+1]);
            read_lines(input_file, line_numbers[i], line_numbers[i+1]);
            printf("boop\n");
        }

        *params_array[i] = ising_model_config{
            .model_id = model_id,
            .num_concurrent = num_concurrent,
            .size = {size[0], size[1]},
            .iterations = iterations,
            .iter_per_step = iter_per_step,
            .seed = seed,
            .temperature = temperature
        };
        
    }

    // Close the file and free the memory
    fclose(input_file);
}
