#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <double.h>
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
        printf("Usage : start step total_steps total_sample \n");
        exit(EXIT_FAILURE);
    }
    
    int start, step, n_tot, tot_sample;

    start = atoi(argv[1]); // Cluster size to start from
    step = atoi(argv[2]); // Size of step
    n_tot = atoi(argv[3]); // Number of steps to made/number of clusters to be sampled
    tot_sample = atoi(argv[4]); // Total number of samples

    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    // Set filenames
    const char *filename1 = "index.bin";
    const char *filename2 = "clusters_to_commitor.bin";
    const char *filename3 = "gridstates.bin";

    // Open file to read
    FILE *ptr1 = fopen(filename1,"rb");
    if (ptr1==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename1);
        exit(EXIT_FAILURE);
    }

    // Creat file to write (or overwrite if file exists)
    FILE *ptr2 = fopen(filename2,"wb");
    if (ptr2==NULL){
        fprintf(stderr,"Error opening %s for write!\n",filename2);
        exit(EXIT_FAILURE);
    }

    // Open file to read
    FILE *ptr3 = fopen(filename3,"rb");
    if (ptr3==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename3);
        exit(EXIT_FAILURE);
    }

    // Create loop variables
    int i = 0, j = 0;
    int cluster_size = 0;
    int tot_uni_clust = 0;

    // Create cluster search array
    int *cluster_search = (int *)malloc((L*L)*sizeof(int));
    if (cluster_search==NULL){
        fprintf(stderr,"Error allocating memory for cluster search!\n");
        exit(EXIT_FAILURE);
    }
 
    // Loops to create cluster size histogram (bins of cluster size with number of occurences of cluster size)
    /*======================================================================================================*/
    // Loop to count how many different cluster sizes exist
    for (i = 0; i < n_grids*n_sweeps/100; i++) { // Divide by 100 due to snapshots being every 100 sweeps
        fseek(ptr1, 8, SEEK_CUR);
        fread(&cluster_size, sizeof(int), 1, ptr1);
        if (cluster_size != 0) {
            cluster_search[cluster_size-1] += 1; // THIS MEANS THAT CLUSTER_SEARCH[0] IS CLUSTER SIZE 1
        }
    }

    // Count number of unique cluster sizes which are non-zero => gives total number of unique cluster sizes
    for (i = 0; i < L*L; i++) {
        if (cluster_search[i] != 0) {
            //printf("%d %d\n",i,cluster_search[i]);
            tot_uni_clust += 1;
        }
    }
    // Create cluster histogram array
    int *cluster_hist = (int *)malloc(L*L*2*sizeof(int)); // tot_uni_clust multiplied by two to store cluster size and number of occurences of cluster size
    if (cluster_hist==NULL){
        fprintf(stderr,"Error allocating memory for cluster histogram!\n");
        exit(EXIT_FAILURE);
    }

    // Initialise cluster_hist
    for (i = 0; i < L*L*2; i++) {cluster_hist[i] = 0;}

    // Populate cluster histogram array
    j = 0;
    for (i = 0; i < L*L*2; i += 2) {
        cluster_hist[i] = i/2+1;
        cluster_hist[i+1] = cluster_search[i/2];
    }
    /*======================================================================================================*/

    // Create rejection sampling probability array
    double *reject_prob = (double *)malloc(L*L*2*sizeof(double)); // tot_uni_clust multiplied by two to store cluster size and probability
    if (reject_prob==NULL){
        fprintf(stderr,"Error allocating memory for rejection probability!\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < L*L*2; i++) {reject_prob[i] = 0.0;} // Initialise probabilities to zero

    double inv_sum = 0.0;
    for (i = 0; i < L*L*2; i += 2) {
        if (cluster_hist[i+1] != 0) {inv_sum += 1.0/(double)cluster_hist[i+1];} // Summing inverse of all cluster occurences
    }

    // Allocating values to reject_prob array
    for (i = 0; i < L*L*2; i += 2) {
        reject_prob[i] = cluster_hist[i];
        if ( cluster_hist[i+1] != 0) {reject_prob[i+1] = (1.0/(double)cluster_hist[i+1])/inv_sum;}
    }

    // Create rejection sampling probability array
    int *sample_selection = (int *)malloc(n_tot*sizeof(int)); // stores how many of each cluster size are chosen to be sampled
    if (sample_selection==NULL){
        fprintf(stderr,"Error allocating memory for sample selection!\n");
        exit(EXIT_FAILURE);
    }

    for ( i = 0; i < n_tot; i++ ) {sample_selection[i] = 0;} // initialise sample selection array to zero

    // Sampling loop, gives how many of each cluster size is to be sampled
    j = 0;
    double random;
    int cutoff = 0;
    while (j < tot_sample) {
        if (cutoff = 2*tot_sample) {break;} // Prevents from loop endelssly running
        for (i = start; i < n_tot*step+start; i += step) {
            if ( j == tot_sample ) {break;}
            random = (double)rand()/(double)(RAND_MAX);
            if ( random < reject_prob[i+1] && sample_selection[(i-start)/step] < cluster_hist[(i-1)*2+1]) { // If random is less than rejection probability then select cluster for sampling
                                                                                                            // Second part of if ensure that you aren't trying to sample more clusters than exist of that size
                sample_selection[(i-start)/step] += 1; // Fancy index simply turns for loop index into a index that starts at 0 and increments by 1 (i = 0; i++)
                j += 1;
            }
        }
        cutoff += 1;
    }
    // Use sample selection to get ngrids and nsweep from the index_cluster.bin, this will then be used to get ising snapshots for commitor calculation
    // Input the data into write array

    // Create temp storage array
    int *temp_store = (int *)malloc(n_grids*2*sizeof(int));
    if (temp_store==NULL){
        fprintf(stderr,"Error allocating memory for temporary storage!\n");
        exit(EXIT_FAILURE);
    }

    // Create temp array to store whether snapshot has been selected
    int *temp_select = (int *)malloc(n_grids*2*sizeof(int));
    if (temp_select==NULL){
        fprintf(stderr,"Error allocating memory for temporary selection!\n");
        exit(EXIT_FAILURE);
    }

    for ( i = 0; i < n_grids*2; i++ ) {temp_select[i] = 0; temp_store[i] = 0;} // Initialise temp_select and temp_store array to zero

    int k = 0, r = 0, sample_value = 0, progress = 0;

    for (j = start; j < step*n_tot+start; j += step) {
        fseek(ptr1, 0, SEEK_SET); // Seek to start of file
        sample_value = sample_selection[(j-start)/step];
        if ( sample_value == 0 ) {continue;} // if cluster size is zero, skip
        k = 0;
        for (i = 0; i < n_grids*n_sweeps/100; i++) { // Divide by 100 due to snapshots being every 100 sweeps
            fseek(ptr1, 8, SEEK_CUR); // Move forward two integers
            fread(&cluster_size, sizeof(int), 1, ptr1); // Read cluster size
            if (cluster_size == j) {
                fseek(ptr1, -12, SEEK_CUR); // Move back three integers
                fread(&temp_store[k], sizeof(int), 1, ptr1); // Read sweep
                fread(&temp_store[k+1], sizeof(int), 1, ptr1); // Read grid
                fseek(ptr1, 4, SEEK_CUR); // Move forward one integer to return to position before moving back
                k += 2;         
            }
        }
        i = 0;
        while (i < sample_value) {
            r = rand()%(k/2); // Random number between 0 and k
            if (temp_select[r] == 0) { // Check if this snapshot has been chosen
                temp_select[r] = 1;
                progress += 1;
                printf("\rNumber of clusters written: %d/%d", progress, tot_sample); // Print progress
                fflush(stdout);
                fwrite(&temp_store[r*2], sizeof(int), 1, ptr2); // Write sweep
                fwrite(&temp_store[r*2+1], sizeof(int), 1, ptr2); // Write grid
                fwrite(&j, sizeof(int), 1, ptr2); // Write cluster size
                i += 1;
            }
        }
        for (i = 0; i < n_grids*2; i++) {temp_select[i] = 0; temp_store[i] = 0;} // Reset temp_select to zero
    }
    printf("\n");

    free(cluster_search); free(cluster_hist); free(reject_prob); free(sample_selection); free(temp_store); free(temp_select);
    fclose(ptr1); fclose(ptr2); fclose(ptr3);

    return EXIT_SUCCESS;
}  
