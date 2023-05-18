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
    if (argc != 5) {
        printf("Usage : start stop divisions samples_per_division\n");
        exit(EXIT_FAILURE);
    }
    
    int start, stop, divisions, samples_per_division;

    start = atoi(argv[1]); // Cluster size to start from
    stop = atoi(argv[2])+1; // Cluster size to end at, +1 in order to make range inclusive
    divisions = atoi(argv[3]); // How sample range is split i.e. 2 means sample range is divided in half
    samples_per_division = atoi(argv[4]); // Samples to be chosen per division

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

    // Create cluster histogram array
    int *cluster_hist = (int *)malloc(L*L*sizeof(int)); // tot_uni_clust multiplied by two to store cluster size and number of occurences of cluster size
    if (cluster_hist==NULL){
        fprintf(stderr,"Error allocating memory for cluster histogram!\n");
        exit(EXIT_FAILURE);
    }

    // Initialise cluster_hist
    for (i = 0; i < L*L; i++) {cluster_hist[i] = 0;}
    
    fseek(index_file, 16, SEEK_CUR);
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            fread(&cluster_size_db, sizeof(double), 1, index_file);
            fseek(index_file, 24, SEEK_CUR); // Skip to next cluster size entry
            cluster_size_int = (int)cluster_size_db;
            cluster_hist[cluster_size_int] += 1;
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

    // Check if samples exist, if not end program
    if (minimum_div_samples == 0) {
        printf("Number of minimum samples is zero. Input different range executing program.\n");
        exit(0);
    }

    // Process commandline input for number of samples per division
    //printf("Minimum number of samples across divisions: %d.\nInput number of samples per divisons, number must be smaller or equal to minimum:\n", minimum_div_samples);
    //fflush(stdout);
    //scanf("%d", &samples_per_division);

    // Create rejection sampling probability array
    double *reject_prob = (double *)malloc(L*L*sizeof(double)); // tot_uni_clust multiplied by two to store cluster size and probability
    if (reject_prob==NULL){
        fprintf(stderr,"Error allocating memory for rejection probability!\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < L*L; i++) {reject_prob[i] = 0.0;} // Initialise probabilities to zero

    double inv_sum = 0.0;
    for (i = 0; i < L*L; i++) {
        if (cluster_hist[i] != 0) {inv_sum += 1.0/(double)cluster_hist[i];} // Summing inverse of all cluster occurences
    }

    // Allocating values to reject_prob array
    for (i = 0; i < L*L; i++) {
        if ( cluster_hist[i] != 0) {reject_prob[i] = (1.0/(double)cluster_hist[i])/inv_sum;}
    }

    int *sample_selection = (int *)malloc(divisions*samples_per_division*sizeof(int)); // stores how many of each cluster size are chosen to be sampled
    if (sample_selection==NULL){
        fprintf(stderr,"Error allocating memory for sample selection!\n");
        exit(EXIT_FAILURE);
    }

    int maximum_random = nsweeps/100*nreplicas; // Number of rows within index.bin file i.e. number of individual grid entries
    int random_selection = 0; // Used to randomly select grid from index.bin file
    double random_percentage = 0.0; // Used to generate a number between 0 and 1 for sample acceptance
    int bytes_per_row = 4*8; // 4 entries per row times 8 bytes per entry (doubles)
    int bytes_to_cluster_size = 2*8; // Cluster size entry is third entry, hence skip 2 entries of size 8 bytes (double)
    double selected_cluster_size = 0.0;

    // Create output array for ngrid, islice, cluster and spare for commitor
    double *output = (double *)malloc(4*sizeof(double));
    if (output==NULL){
        fprintf(stderr,"Error allocating memory for output array!\n");
        exit(EXIT_FAILURE);
    } 

    i = 0;
    while(i < divisions*samples_per_division) {
        random_selection = rand()%maximum_random;
        random_percentage = (double)rand()/(double)(RAND_MAX);
        fseek(index_file, random_selection*bytes_per_row+bytes_to_cluster_size, SEEK_SET); // Seek from start of index file to currently selected grid cluster size
        fread(&selected_cluster_size, sizeof(double), 1, index_file);
        if (random_percentage < reject_prob[(int)selected_cluster_size]) {
            fseek(index_file, -24, SEEK_CUR); // Seek back 24 bytes (3 doubles) to the start of the index.bin row
            fread(&output[0], sizeof(double), 1, index_file);
            fread(&output[1], sizeof(double), 1, index_file);
            fread(&output[2], sizeof(double), 1, index_file);
            fread(&output[3], sizeof(double), 1, index_file);
            fwrite(output, sizeof(double), 4, commitor_file);
            i += 1;
        }
    }  

    //printf("\n"); // Newline for command prompt

    free(cluster_search); free(cluster_hist); free(reject_prob); free(sample_selection); free(output); free(division_samples);
    fclose(index_file); fclose(commitor_file); fclose(gridstates_file);

    /*
    FILE *commitor_file1 = fopen(commitor_filename,"rb");
    if (commitor_file1==NULL){
        fprintf(stderr,"Error opening %s for write!\n",commitor_filename);
        exit(EXIT_FAILURE);
    }

    fseek(commitor_file1, 0, SEEK_SET);
    for (i=0;i<samples_per_division*4;i++) {
        if (i%4==0) {printf("\n");}
        fread(&random_percentage, sizeof(double), 1, commitor_file1);
        printf("%f ", random_percentage);
    }

    fclose(commitor_file1);
    */

    printf("Samples successfully chosen \n");
    return EXIT_SUCCESS;
}  
