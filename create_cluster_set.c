#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "functions/read_input_variables.h" 
#include "functions/comparison.h" 

#define MOD(a,b) ((((a)%(b))+(b))%(b)) // Custom definition so that mod works correctly with negative numbers

int main (int argc, char *argv[]) {
    
    // Setup for timing code
    clock_t start, end;
    double execution_time;
    start = clock();

    // Initialise pseudo-random number generation
    srand(time(NULL));

    // Process commandline input
    if (argc != 4) {
        printf("Usage : samples_per_cluster_size min_cluster_size max_cluster_size\n");
        printf("Set either min_cluster_size or max_cluster_size to -1 to use default value.\n");
        exit(EXIT_FAILURE);
    }
    
    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    int samples;
    int min_cluster_size = 50;
    int max_cluster_size = (int)((double)(L*L)*0.95);

    samples = atoi(argv[1]); // Number of samples to be chosen
    if (atoi(argv[2]) != -1) {
        min_cluster_size = atoi(argv[2]);
    }
    if (atoi(argv[2]) != -1) {
        max_cluster_size = atoi(argv[3]);
    }

    // Set filenames
    const char *index_filename = "index.bin";
    const char *committor_filename = "committor_index.bin";

    // Open file to read
    FILE *index_file = fopen(index_filename,"rb");
    if (index_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", index_filename);
        exit(EXIT_FAILURE);
    }

    // Remove committor file
    remove(committor_filename);
    // Create file to write
    FILE *committor_file = fopen(committor_filename,"wb");
    if (committor_file==NULL){
        fprintf(stderr,"Error opening %s for write!\n",committor_filename);
        exit(EXIT_FAILURE);
    }
    
    // Create loop variables
    int i = 0, j = 0, k = 0;
    int islice, igrid;
    int cluster_size = 0;

    // Create cluster histogram array
    int *cluster_hist = (int *)malloc(L*L*sizeof(int)); // tot_uni_clust multiplied by two to store cluster size and number of occurences of cluster size
    if (cluster_hist==NULL){
        fprintf(stderr,"Error allocating memory for cluster histogram!\n");
        exit(EXIT_FAILURE);
    }

    // Create arrays for storing for ngrid, islice, cluster and spare for committor        
    int *store_ngrid = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_ngrid==NULL){fprintf(stderr,"Error allocating memory for store_ngrid array!\n"); exit(EXIT_FAILURE);} 
    int *store_slice = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_slice==NULL){fprintf(stderr,"Error allocating memory for store_slice array!\n"); exit(EXIT_FAILURE);} 
    int *store_cluster = (int *)malloc(nreplicas*nsweeps/100*sizeof(int));
    if (store_cluster==NULL){fprintf(stderr,"Error allocating memory for store_cluster array!\n"); exit(EXIT_FAILURE);} 
    double *store_committor = (double *)malloc(nreplicas*nsweeps/100*sizeof(double));
    if (store_committor==NULL){fprintf(stderr,"Error allocating memory for store_committor array!\n"); exit(EXIT_FAILURE);} 

    // Read and sotre data from index file
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

    // Sort the loaded arrays based on the cluster size
    int **p_store_cluster = malloc(nreplicas*nsweeps/100*sizeof(long));;
    int ta, tb, tc, td;

    // create array of pointers to store_cluster
    for (i = 0; i < nreplicas*nsweeps/100; i++) {
        p_store_cluster[i] = &store_cluster[i];
    }

    // sort array of pointers
    qsort(p_store_cluster, nreplicas*nsweeps/100, sizeof(p_store_cluster[0]), compare);
    
    // reorder loaded arrays according to the array of pointers
    for(i=0;i<nreplicas*nsweeps/100;i++){
        if(i != p_store_cluster[i]-store_cluster){
            ta = store_ngrid[i];
            tb = store_slice[i];
            tc = store_cluster[i];
            td = store_committor[i];
            k = i;
            while(i != (j = p_store_cluster[k]-store_cluster)){
                store_ngrid[k] = store_ngrid[j];
                store_slice[k] = store_slice[j];
                store_cluster[k] = store_cluster[j];
                store_committor[k] = store_committor[k];
                p_store_cluster[k] = &store_cluster[k];
                k = j;
            }
            store_ngrid[k] = ta;
            store_slice[k] = tb;
            store_cluster[k] = tc;
            store_committor[k] = td;
            p_store_cluster[k] = &store_cluster[k];
        }
    }

    free(p_store_cluster); // Freed since no longer needed

    // Create array for storing starting index of each cluster size and how many of a given cluster exist
    int *store_cluster_index = (int *)malloc(64*64*2*sizeof(int));
    if (store_cluster_index==NULL){fprintf(stderr,"Error allocating memory for store_cluster_index array!\n"); exit(EXIT_FAILURE);}
    for (i=0;i<64*64*2;i++) {store_cluster_index[i]=0;}

    int cluster_check = 0;
    cluster_size = 0;
    for (i=0;i<nreplicas*nsweeps/100;i++) {
        cluster_size = store_cluster[i];
        if ( cluster_size > cluster_check) {
            cluster_check = cluster_size;
            store_cluster_index[cluster_size*2] = i;
        }
        store_cluster_index[cluster_size*2+1] = store_cluster_index[cluster_size*2+1]+1;
    }


    // Initialise cluster_hist
    for (i = 0; i < L*L; i++) {cluster_hist[i] = 0;}
    // Loops over slices
    for (islice=0;islice<nsweeps/100;islice++) {
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            cluster_size = store_cluster[igrid+nreplicas*islice];
            cluster_hist[cluster_size] += 1;
        } // igrid
    } // isweep

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
    int bins = 250;
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

    int random_selection = 0; // Used to randomly select grid from index.bin file
    int selected_cluster_size = 0;

    int counter = 0;
    for (i=min_cluster_size;i<max_cluster_size+1;i++) {
        j = 0;
        while(j < samples) {
            if (store_cluster_index[i*2+1] == 0) {break;}
            random_selection = rand()%store_cluster_index[i*2+1]+store_cluster_index[i*2];
            selected_cluster_size = store_cluster[random_selection];
            j += 1;
            fwrite(&store_slice[random_selection], sizeof(int), 1, committor_file);
            fwrite(&store_ngrid[random_selection], sizeof(int), 1, committor_file);
            fwrite(&selected_cluster_size, sizeof(int), 1, committor_file);
            fwrite(&store_committor[random_selection], sizeof(double), 1, committor_file);
            fwrite(&store_committor[random_selection], sizeof(double), 1, committor_file); // Write to create space for standard deviation
        }
        counter += 1;
        printf("\rPercentage of samples selected: %d%%", (int)((double)counter/(double)((max_cluster_size-min_cluster_size+1))*100)); // Print progress
        fflush(stdout);
    }  

    printf("\n"); // Newline

    free(cluster_hist); free(reject_prob);
    free(store_ngrid); free(store_slice); free(store_cluster); free(store_committor); free(binned_prob);
    fclose(index_file); fclose(committor_file);

    // Print time taken for program to execute
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds \n", execution_time);
    
    return EXIT_SUCCESS;
}  
