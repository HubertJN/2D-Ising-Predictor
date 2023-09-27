#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "functions/read_input_variables.h"  

int main (int argc, char *argv[]) {
    
    // Setup for timing code
    clock_t start, end;
    double execution_time;
    start = clock();

    // Initialise pseudo-random number generation
    srand(time(NULL));

    // Process commandline input
    if (argc != 2) {
        printf("Usage : evo_index\n");
        printf("Set evo_index to -1 to choose random lattice.\n");
        exit(EXIT_FAILURE);
    }
    
    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    int evo_index = rand()%nreplicas;

    if (atoi(argv[1]) != -1) {
        evo_index = atoi(argv[1]);
    }

    // Set filenames
    const char *index_filename = "index.bin";
    const char *evolution_filename = "evolution_index.bin";

    // Open file to read
    FILE *index_file = fopen(index_filename,"rb");
    if (index_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", index_filename);
        exit(EXIT_FAILURE);
    }

    // Remove evolution file
    remove(evolution_filename);
    // Create file to write
    FILE *evolution_file = fopen(evolution_filename,"wb");
    if (evolution_file==NULL){
        fprintf(stderr,"Error opening %s for write!\n",evolution_filename);
        exit(EXIT_FAILURE);
    }
    
    // Create loop variables
    int islice, igrid;

    // Create arrays for storing for ngrid, islice, cluster and spare for committor        
    int *store_ngrid = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_ngrid==NULL){fprintf(stderr,"Error allocating memory for store_ngrid array!\n"); exit(EXIT_FAILURE);} 
    int *store_slice = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_slice==NULL){fprintf(stderr,"Error allocating memory for store_slice array!\n"); exit(EXIT_FAILURE);} 
    int *store_cluster = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_cluster==NULL){fprintf(stderr,"Error allocating memory for store_cluster array!\n"); exit(EXIT_FAILURE);} 
    double *store_committor = (double *)malloc(nreplicas*nsweeps/100*sizeof(double));
    if (store_committor==NULL){fprintf(stderr,"Error allocating memory for store_committor array!\n"); exit(EXIT_FAILURE);} 

    // Loops over slices
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            fread(&store_slice[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_ngrid[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_cluster[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_committor[igrid+nreplicas*islice], sizeof(double), 1, index_file);
            fread(&store_committor[igrid+nreplicas*islice], sizeof(double), 1, index_file); // Dummy read that reads the standard deviation value in, which at this stage is also -1
        }
    }

    for (islice=0;islice<nsweeps/100;islice++) {
        fwrite(&store_slice[evo_index+nreplicas*islice], sizeof(int), 1, evolution_file);
        fwrite(&store_ngrid[evo_index+nreplicas*islice], sizeof(int), 1, evolution_file);
        fwrite(&store_cluster[evo_index+nreplicas*islice], sizeof(int), 1, evolution_file);
        fwrite(&store_committor[evo_index+nreplicas*islice], sizeof(double), 1, evolution_file);
        fwrite(&store_committor[evo_index+nreplicas*islice], sizeof(double), 1, evolution_file); // Write to create space for standard deviation
        if (store_cluster[evo_index+nreplicas*islice] > 4000) {
            break;
        }
    }

    printf("Selection complete.\n");

    fclose(index_file); fclose(evolution_file);
    free(store_ngrid); free(store_slice); free(store_cluster); free(store_committor);

    // Print time taken for program to execute
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds \n", execution_time);
    
    return EXIT_SUCCESS;
}  
