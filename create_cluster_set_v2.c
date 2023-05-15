#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "functions/functions.h"

#define MOD(a,b) ((((a)%(b))+(b))%(b))

int main (int argc, char *argv[]) {
    
    // Initialise pseudo-random number generation
    srand(time(NULL));

    // Process commandline input
    if (argc != 4) {
        printf("Usage : start stop divisions \n");
        exit(EXIT_FAILURE);
    }
    
    int start, stop, divisions, samples_per_division;

    start = atoi(argv[1]); // Cluster size to start from
    stop = atoi(argv[2])+1; // Cluster size to end at, +1 in order to make range inclusive
    divisions = atoi(argv[3]); // How sample range is split i.e. 2 means sample range is divided in half

    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    // Set filenames
    const char *index_filename = "index.bin";
    const char *commitor_filename = "commitor_calc_index.bin";
    const char *gridstates_filename = "gridstates.bin";

    // Open file to read
    FILE *index_file = fopen(index_filename,"rb");
    if (index_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", index_filename);
        exit(EXIT_FAILURE);
    }

    // Remove commitor file
    remove(commitor_filename);
    // Create file to write
    FILE *commitor_file = fopen(commitor_filename,"wb");
    if (commitor_file==NULL){
        fprintf(stderr,"Error opening %s for write!\n",commitor_filename);
        exit(EXIT_FAILURE);
    }

    // Open file to read
    FILE *gridstates_file = fopen(gridstates_filename,"rb");
    if (gridstates_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", gridstates_filename);
        exit(EXIT_FAILURE);
    }

    // Create loop variables
    int i = 0, j = 0;
    int islice, igrid, idiv;
    double cluster_size_db = 0.0;
    int cluster_size_int = 0;
    int tot_uni_clust = 0;

    // Create cluster search array
    int *cluster_search = (int *)malloc((L*L)*sizeof(int));
    if (cluster_search==NULL){
        fprintf(stderr,"Error allocating memory for cluster search!\n");
        exit(EXIT_FAILURE);
    }
    
    // Create divisions
    double division_range = (stop - start)/divisions;

    // Check minimum number of available samples per division
    // Create sample number per division array and initialise
    int *division_samples = (int *)malloc(divisions*sizeof(int));
    if (division_samples==NULL){
        fprintf(stderr,"Error allocating memory for division samples!\n");
        exit(EXIT_FAILURE);
    }
    for (idiv=0;idiv<divisions;idiv++) {division_samples[idiv] = 0;}

    fseek(index_file, 0, SEEK_SET); // Skip first two doubles

    for (islice=0;islice<120;islice++) {
        if (islice%4 == 0) {printf("\n");}
        fread(&cluster_size_db, sizeof(double), 1, index_file);
        cluster_size_int = (int)cluster_size_db;
        printf("%d ", cluster_size_int);
        
    }
    printf("\n");

/*
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            fread(&cluster_size_int, sizeof(int), 1, index_file);
            fseek(index_file, 0, SEEK_CUR); // Skip to next cluster size entry
            cluster_size_int = (int)cluster_size_db;
            if (igrid==0) {printf("%d ", cluster_size_int);}
            // Loop to check if cluster size fits in divisions, +1 to stop variables plays a role here in order to include stop within sampling
            // When "cluster_size_int < start+(idiv+1)*division_range" is checked, stop is included since start+(idiv+1)*division_range is 1 higher than stop cluster size
            for (idiv=0;idiv<divisions;idiv++) {
                if  (cluster_size_int > start+idiv*division_range-1 && cluster_size_int < start+(idiv+1)*division_range) {
                    division_samples[idiv] += 1;
                }
            }
        } // igrid
    } // isweep
    
    // Find minimum number of sample across divisions
    int minimum_div_samples = division_samples[0];
    for (idiv=0;idiv<divisions;idiv++) {
        if (minimum_div_samples < division_samples[idiv]) {
            minimum_div_samples = division_samples[idiv];
        }
    }
    printf("\n");
    // Check if samples exist, if not end program
    if (minimum_div_samples == 0) {
        printf("Number of minimum samples is zero. Input different range executing program.\n");
        exit(0);
    }

    // Process commandline input for number of samples per division
    printf("Minimum number of samples in across divisons: %d.\nInput number of samples per divisons, number must be smaller or equal to minimum:\n", minimum_div_samples);
    fflush(stdout); 
    scanf("%d", samples_per_division);

    */
    return EXIT_SUCCESS;
}  
