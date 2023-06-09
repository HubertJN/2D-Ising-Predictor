#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "functions/read_input_variables.h" 

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
    int i = 0, j = 0, k = 0;
    int islice, igrid, idiv;
    int cluster_size = 0;

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

    // Create arrays for storing for ngrid, islice, cluster and spare for commitor        
    int *store_ngrid = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_ngrid==NULL){fprintf(stderr,"Error allocating memory for store_ngrid array!\n"); exit(EXIT_FAILURE);} 
    int *store_slice = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_slice==NULL){fprintf(stderr,"Error allocating memory for store_slice array!\n"); exit(EXIT_FAILURE);} 
    int *store_cluster = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_cluster==NULL){fprintf(stderr,"Error allocating memory for store_cluster array!\n"); exit(EXIT_FAILURE);} 
    double *store_commitor = (double *)malloc(nreplicas*nsweeps/100*sizeof(double));
    if (store_commitor==NULL){fprintf(stderr,"Error allocating memory for store_commitor array!\n"); exit(EXIT_FAILURE);} 

    // Read and sotre data from index file
    // Loops over slices
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            fread(&store_ngrid[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_slice[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_cluster[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_commitor[igrid+nreplicas*islice], sizeof(double), 1, index_file);
        }
    }

    // Initialise cluster_hist
    for (i = 0; i < L*L; i++) {cluster_hist[i] = 0;}
    // Loops over slices
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            cluster_size = store_cluster[igrid+nreplicas*islice];
            cluster_hist[cluster_size] += 1;
            // Loop to check if cluster size fits in divisions, +1 to stop variables plays a role here in order to include stop within sampling
            // When "cluster_size_int < start+(idiv+1)*division_range" is checked, stop is included since start+(idiv+1)*division_range is 1 higher than stop cluster size
            for (idiv=0;idiv<divisions;idiv++) {
                if  (cluster_size > start+idiv*division_range-1 && cluster_size < start+(idiv+1)*division_range) {
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
        printf("Number of minimum samples is zero. Input different range when executing program.\n");
        exit(0);
    }

    // Process commandline input for number of samples per division
    //printf("Minimum number of samples across divisions: %d.\nInput number of samples per divisons, number must be smaller or equal to minimum:\n", minimum_div_samples);
    //fflush(stdout);
    //scanf("%d", &samples_per_division);

    // Create rejection sampling probability array
    double *reject_prob = (double *)malloc(L*L*sizeof(double)); // tot_uni_clust multiplied by two to store cluster size and probability
    if (reject_prob==NULL){
        fprintf(stderr,"Error allocating memory for rejection probability array!\n");
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

    // Create an array of binned statistics
    int bins = 50;
    int bin_remainder = 0;
    int bin_size = 0;
    double bin_sum = 0.0;
    double *binned_prob = (double *)malloc(bins*sizeof(double)); // tot_uni_clust multiplied by two to store cluster size and probability
    if (binned_prob==NULL){
        fprintf(stderr,"Error allocating memory for binned probability array!\n");
        exit(EXIT_FAILURE);
    }

    bin_remainder = MOD(L*L, bins);
    bin_size = (L*L-bin_remainder)/bins;

    k = 0;

    // Loop through first bins up to the remainder after dividing by bins. Each bin size is increased by size 1
    for (i = 0; i < bin_remainder; i++) {
        bin_sum = 0.0;
        for (j = 0; j < bin_size+1; j++) {
            bin_sum += reject_prob[j+k];
        }
        binned_prob[i] = bin_sum;
        k += bin_size+1;
    }
    // Loop through last bins with bin size as normal
    for (i = bin_remainder; i < bins; i++) {
        bin_sum = 0.0;
        for (j = 0; j < bin_size; j++) {
            bin_sum += reject_prob[j+k];
        }
        binned_prob[i] = bin_sum;
        k += bin_size;
    }

    // Set the rejection probability array to use the binned values for ease of calculation later
    k = 0;
    for (i = 0; i < bin_remainder; i++) {
        for (j = 0; j < bin_size+1; j++) {
            reject_prob[j+k] = binned_prob[i];
        }
        k += bin_size+1;
    }
    for (i = bin_remainder; i < bins; i++) {
        for (j = 0; j < bin_size; j++) {
            reject_prob[j+k] = binned_prob[i];
        }
        k += bin_size;
    }
    int *sample_selection = (int *)malloc(divisions*samples_per_division*sizeof(int)); // stores how many of each cluster size are chosen to be sampled
    if (sample_selection==NULL){
        fprintf(stderr,"Error allocating memory for sample selection!\n");
        exit(EXIT_FAILURE);
    }

    int maximum_random = nsweeps/100*nreplicas; // Number of rows within index.bin file i.e. number of individual grid entries
    int random_selection = 0; // Used to randomly select grid from index.bin file
    double random_percentage = 0.0; // Used to generate a number between 0 and 1 for sample acceptance
    int selected_cluster_size = 0;

    i = 0;
    while(i < divisions*samples_per_division) {
        random_selection = rand()%maximum_random;
        random_percentage = (double)rand()/(double)(RAND_MAX);
        selected_cluster_size = store_cluster[random_selection];
        if (random_percentage < reject_prob[selected_cluster_size]) {
            fwrite(&store_ngrid[random_selection], sizeof(int), 1, commitor_file);
            fwrite(&store_slice[random_selection], sizeof(int), 1, commitor_file);
            fwrite(&selected_cluster_size, sizeof(int), 1, commitor_file);
            fwrite(&store_commitor[random_selection], sizeof(double), 1, commitor_file);
            i += 1;
        }
    }  

    //printf("\n"); // Newline for command prompt

    free(cluster_search); free(cluster_hist); free(reject_prob); free(sample_selection); 
    free(store_ngrid); free(store_slice); free(store_slice); free(store_commitor); free(division_samples); free(binned_prob);
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
