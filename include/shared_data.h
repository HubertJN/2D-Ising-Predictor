#ifndef SHARED_DATA_H
#define SHARED_DATA_H


const int n_dims = 2;

// structure to hold the configuration of the ising model, not all parameters are used in all models
// NOTE: when adding new user set parameters update the read_input_file function in input_reader.cu
typedef struct ising_model_config {
    // User set parameters in config file
    int model_id;     // model id
    int num_concurrent; // number of concurrent simulations
    int size[n_dims];         // size of each grid, 2D
    int iterations;   // number of iterations in the simulation
    int iter_per_step; // number of iterations per step
    int seed;         // seed for the random number generator
    float inv_temperature;  // temperature of the system
    float field;        // magnetic field strength
    char* input_file; // input file name
    int starting_config; // starting configuration, 0 for file input, 1 for random, 2 for all up, 3 for all down
    float nucleation_threshold; // nucleation threshold
    // User or System set parameters
    int num_threads;  // number of threads per block
    // System set parameters
    int num_blocks;   // number of blocks
    int element_size; // size of the grid elements in bytes
    size_t mem_size;     // size of all grids required in bytes
    int prob_size;    // size of the probability array
} ising_model_config;

#endif
