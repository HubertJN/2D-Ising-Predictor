
#include "../include/input_reader.h"

void read_lines(FILE* input_file, int start_line, int end_line) {
    char* line;
    size_t len = 0;
    int line_num = 0;
    // Reset the file to begining
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

    // throwaway variables for kvp
    int model_id = -1;     // model id
    int model_itask = -1;  // model itask
    int num_concurrent = -1; // number of concurrent simulations
    int size[2]; size[0] = 0; size[1] = 0;         // size of each grid dimension
    int L = 0;              // size of grid L == size[0] = size[1])
    int iterations = 0;   // number of iterations in the simulation
    int iter_per_step = 0; // number of iterations per step
    int seed = 0;         // seed for the random number generator
    float field = 0.0;       // external field
    float inv_temperature = 0.0; // inverse temperature
    int starting_config = 0.0; // starting configuration
    float nucleation_threshold = 0.0; // nucleation threshold
    float dn_threshold = 0.0;         // Magnetisation at which we consider the system to have reached spin up state
    float up_threshold = 0.0;         // Magnetisation at which we consider the system to have reached spin up state
    int num_threads;
    char grid_file[grid_file_str_len];

    kvp_register_i("model_id", &model_id);
    kvp_register_i("model_itask", &model_itask);
    kvp_register_i("num_concurrent", &num_concurrent);
    kvp_register_i("size_x", &size[0]);
    kvp_register_i("size_y", &size[1]);
    kvp_register_i("L", &L); // Not in the struct will be mapped to size[0] and size[1]
    kvp_register_i("iterations", &iterations);
    kvp_register_i("iter_per_step", &iter_per_step);
    kvp_register_i("seed", &seed);
    kvp_register_f("inv_temperature", &inv_temperature);
    kvp_register_f("field", &field);
    kvp_register_i("num_threads", &num_threads);
    kvp_register_string("input_file", grid_file, grid_file_str_len);
    kvp_register_i("starting_config", &starting_config);
    kvp_register_f("nucleation_threshold", &nucleation_threshold);
    
    fprintf(stderr, "found %d models\n", models);
    for (int i = 0; i < models; i++) {

        // reset the kvp throaway variables to default values or fail states
        model_id = -1;
        model_itask = -1;
        num_concurrent = -1;
        size[0] = 0;
        size[1] = 0;
        L = 0;
        iterations = 0;
        iter_per_step = 0;
        seed = 0;
        field = 0.0;
        inv_temperature = 1.0;
        starting_config = -1;
        nucleation_threshold = 0.0;
        dn_threshold = 0.0;
        up_threshold = 0.0;
        strcpy(grid_file, "NONE");



        fprintf(stderr, "Reading model %d\n", i);
        fflush(stderr);
        // get the line number pairs
        if (i == models - 1) {
            // read from line_numbers[i] to eof
            fprintf(stderr, "Reading from line %d to eof\n", line_numbers[i]);
            read_lines(input_file, line_numbers[i], line_num);
        }
        else {
            // read from line_numbers[i] to line_numbers[i+1]
            fprintf(stderr, "Reading from line %d to line %d\n", line_numbers[i], line_numbers[i+1]);
            read_lines(input_file, line_numbers[i], line_numbers[i+1]);
        }

        // Run assertions and calculations on the input parameters
        // Check the model id and task have been set
        if (model_id == -1) {
            fprintf(stderr, "Error: Model id not set\n");
            exit(1);
        }
        if (model_itask == -1) {
            fprintf(stderr, "Error: Model itask not set\n");
            exit(1);
        }
        // Check the number of concurrent simulations has been set
        if (num_concurrent == -1) {
            fprintf(stderr, "Error: Number of concurrent simulations not set\n");
            exit(1);
        }
        // Check the grid size has been set
        if ((size[0] == 0 || size[1] == 0) && L == 0)  {
            // No grid size set
            fprintf(stderr, "Error: Grid size not set\n");
            exit(1);
        }
        if (size[0] == 0 && size[1] == 0 && L != 0) {
            // L set apply to both dimensions
            size[0] = L;
            size[1] = L;
        }
        if ((size[0] == 0 && size[1] != 0) || (size[0] != 0 && size[1] == 0)){
            // L unset and only one dimension set
            fprintf(stderr, "Error:size_x and size_y must be set\n");
            exit(1);
        }

        // Check nucleation thresholds have been set correctly
        if (nucleation_threshold != 0.0) {
            if (up_threshold != 0.0 || dn_threshold != 0.0) {
                fprintf(stderr, "Error: Cannot specify both nucleation_threshold and (up||dn)_threshold\n");
                exit(1);
            }
            up_threshold = nucleation_threshold;
            dn_threshold = -nucleation_threshold;
        } else if (up_threshold == 0.0 || dn_threshold == 0.0) {
            if (up_threshold == 0.0) {
                fprintf(stderr, "Error: Must specify up_threshold \n");
                exit(1);
            }
            if (dn_threshold == 0.0 && model_itask == 0) {
                fprintf(stderr, "Error: Must specify dn_threshold \n");
                exit(1);
            }
            if (model_itask == 0) {
                fprintf(stderr, "Error: Must specify nucleation_threshold if not specifying up_threshold and dn_threshold\n");
                exit(1);
            }
        } else if (up_threshold <= dn_threshold) {
            fprintf(stderr, "Error: up_threshold must be greater than dn_threshold\n");
            exit(1);
        } else if ( up_threshold > 1.0 || dn_threshold < -1.0) {
            fprintf(stderr, "Error: up_threshold must be less than 1.0 and dn_threshold must be greater than -1.0\n");
            exit(1);
        }
        
        // Check the task is valid
        if (model_itask != -1) {
            // For a committer calculation we require a grid from a file
            if (model_itask == 0 && starting_config != 0){
                fprintf(stderr, "Error: Starting config must be 0, load from file, for committer calculation\n");
                exit(1);
            }
            if (starting_config == 0){
                // Check the input file has been set
                if (strcmp(grid_file, "NONE") == 0) {
                    fprintf(stderr, "Error: Input file must be set for starting config 0\n");
                    exit(1);
                }
            }
        }

        // Check the number of iterations has been set
        if (iterations == 0) {
            fprintf(stderr, "Error: Number of iterations not set\n");
            exit(1);
        }
        // Check the number of iterations per step has been set
        if (iter_per_step == 0) {
            fprintf(stderr, "Error: Number of iterations per step not set\n");
            exit(1);
        }

        // Warn on default values
        if (seed == 0) {
            fprintf(stderr, "Warning: Seed not set, using default value, 0\n");
        }
        if (field == 0.0) {
            fprintf(stderr, "Warning: Field not set, using default value, 0\n");
        }
        if (inv_temperature == 0.0) {
            fprintf(stderr, "Warning: Inverse temperature not set, using default value, 1.0\n");
        }
        
        // Save the config into the array
        *params_array[i] = ising_model_config{
            .model_id = model_id,
            .model_itask = model_itask,
            .num_concurrent = num_concurrent,
            .size = {size[0], size[1]},
            .iterations = iterations,
            .iter_per_step = iter_per_step,
            .seed = seed,
            .inv_temperature = inv_temperature,
            .field = field,
            .input_file = grid_file,
            .starting_config = starting_config
        };
    }
    // Close the file and free the memory
    fclose(input_file);
}

void load_grid(cudaStream_t stream, ising_model_config* launch_struct, int* host_grid, int* dev_grid) {
    /* Load the grid from the input file, this function reads a file line by line into host_grid
    then does an async copy to dev_grid
    Parameters:
        launch_struct: struct containing the launch parameters
        host_grid: pointer to the host grid, Pinned memory
        dev_grid: pointer to the device grid
    */

    // Read the grid into host_grid
    readGridsFromFile(launch_struct, host_grid);

    // Copy the grid to dev_grid
    cudaMemcpy(dev_grid, host_grid, launch_struct->size[0] * launch_struct->size[1] * sizeof(int), cudaMemcpyHostToDevice);

    return;
}