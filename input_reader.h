
#ifndef INPUT_READER_H_
#define INPUT_READER_H_

typedef struct ising_model_config {
    int model_id;     // model id
    int num_concurrent; // number of concurrent simulations
    int size[2];         // size of each grid, 2D
    int iterations;   // number of iterations in the simulation
    int iter_per_step; // number of iterations per step
    int seed;         // seed for the random number generator
    double temperature;  // temperature of the system
} ising_model_config;


struct ising_model_config** read_input_file(const char* filename);

#endif /* INPUT_READER_H_ */
