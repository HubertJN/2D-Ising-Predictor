#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include "functions/read_input_variables.h" 
#include "functions/comparison.h" 

#define MOD(a,b) ((((a)%(b))+(b))%(b)) // Custom definition so that mod works correctly with negative numbers

int main (int argc, char *argv[]) {

    /* ---------- Define variables --------- */
    int samples, min_mag, max_mag, min_clust, max_clust; // command line argument variables
    int i, j, k, islice, igrid; // loop variables

    int mag = 0, clust = 0;

    int tot_samples = 0;

    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method; // GPU_Ising input file variables
    double beta, h; // GPU_Ising input file variables

    clock_t start, end;
    double execution_time;

    const char *index_filename = "index.bin";
    const char *committor_filename = "committor_index.bin";
    /* ---------- --------- --------- */

    start = clock();

    srand(time(NULL));

    if (argc != 6) {
        printf("Usage : samples min_mag max_mag min_clust max_clust\n");
        printf("Type :    int     int     int      int      int\n");
        exit(EXIT_FAILURE);
    }
    
    // Define and read input variables
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    samples = atoi(argv[1]);
    min_mag = atoi(argv[2]);
    max_mag = atoi(argv[3]);
    min_clust = atoi(argv[4]);
    max_clust = atoi(argv[5]);

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
    
    // Create arrays for storing for ngrid, islice, cluster and spare for committor        
    int *store_ngrid_tmp = (int *)malloc(nreplicas*nsweeps/grid_output_int*sizeof(int));
    if (store_ngrid_tmp==NULL){fprintf(stderr,"Error allocating memory for store_ngrid_tmp array!\n"); exit(EXIT_FAILURE);} 
    int *store_slice_tmp = (int *)malloc(nreplicas*nsweeps/grid_output_int*sizeof(int));
    if (store_slice_tmp==NULL){fprintf(stderr,"Error allocating memory for store_slice_tmp array!\n"); exit(EXIT_FAILURE);} 
    int *store_mag_tmp = (int *)malloc(nreplicas*nsweeps/grid_output_int*sizeof(int));
    if (store_mag_tmp==NULL){fprintf(stderr,"Error allocating memory for store_mag_tmp array!\n"); exit(EXIT_FAILURE);} 
    int *store_clust_tmp = (int *)malloc(nreplicas*nsweeps/grid_output_int*sizeof(int));
    if (store_clust_tmp==NULL){fprintf(stderr,"Error allocating memory for store_clust_tmp array!\n"); exit(EXIT_FAILURE);} 
    double *store_committor_tmp = (double *)malloc(nreplicas*nsweeps/grid_output_int*sizeof(double));
    if (store_committor_tmp==NULL){fprintf(stderr,"Error allocating memory for store_committor_tmp array!\n"); exit(EXIT_FAILURE);} 

    // Read and store data from index file
    for (islice=0;islice<nsweeps/grid_output_int;islice++) {
        for (igrid=0;igrid<nreplicas;igrid++) {
            fread(&store_slice_tmp[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_ngrid_tmp[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_mag_tmp[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_clust_tmp[igrid+nreplicas*islice], sizeof(int), 1, index_file);
            fread(&store_committor_tmp[igrid+nreplicas*islice], sizeof(double), 1, index_file);
            fread(&store_committor_tmp[igrid+nreplicas*islice], sizeof(double), 1, index_file); // Dummy read that reads the standard deviation value in, which at this stage is also -1
        } // nreplicas
    } // slices

    // Filter out selection based on magnetization range and cluster range
    
    for (i=0;i<nreplicas*nsweeps/grid_output_int;i++) {
        mag = store_mag_tmp[i];
        clust = store_clust_tmp[i];
        if (mag >= min_mag && mag <= max_mag && clust >= min_clust && clust <= max_clust) {
            tot_samples++;
        }
    } // all array elements

    printf("Total available samples within specified range: %d\n", tot_samples);

    // Create arrays to store filtered data   
    int *store_ngrid = (int *)malloc(tot_samples*sizeof(int));
    if (store_ngrid==NULL){fprintf(stderr,"Error allocating memory for store_ngrid array!\n"); exit(EXIT_FAILURE);} 
    int *store_slice = (int *)malloc(tot_samples*sizeof(int));
    if (store_slice==NULL){fprintf(stderr,"Error allocating memory for store_slice array!\n"); exit(EXIT_FAILURE);} 
    int *store_mag = (int *)malloc(tot_samples*sizeof(int));
    if (store_mag==NULL){fprintf(stderr,"Error allocating memory for store_mag array!\n"); exit(EXIT_FAILURE);} 
    int *store_clust = (int *)malloc(tot_samples*sizeof(int));
    if (store_clust==NULL){fprintf(stderr,"Error allocating memory for store_clust array!\n"); exit(EXIT_FAILURE);} 
    double *store_committor = (double *)malloc(tot_samples*sizeof(double));
    if (store_committor==NULL){fprintf(stderr,"Error allocating memory for store_committor array!\n"); exit(EXIT_FAILURE);} 

    j = 0;
    for (i=0;i<nreplicas*nsweeps/grid_output_int;i++) {
        mag = store_mag_tmp[i];
        clust = store_clust_tmp[i];
        if (mag >= min_mag && mag <= max_mag && clust >= min_clust && clust <= max_clust) {
            store_ngrid[j] = store_ngrid_tmp[i];
            store_slice[j] = store_slice_tmp[i];
            store_mag[j] = mag;
            store_clust[j] = clust;
            store_committor[j] = -1.0;
            j++;
        }
    }


    // Sort the loaded arrays based on the magnetization
    int **p_store_mag = malloc(tot_samples*sizeof(long));
    int ta, tb, tc, td, te;
    // create array of pointers to store_cluster
    for (i = 0; i < tot_samples; i++) {
        p_store_mag[i] = &store_mag[i];
    }

    // sort array of pointers
    qsort(p_store_mag, tot_samples, sizeof(long), compare);
    
    // reorder loaded arrays according to the array of pointers
    for(i=0;i<tot_samples;i++){
        if(i != p_store_mag[i]-store_mag){
            ta = store_ngrid[i];
            tb = store_slice[i];
            tc = store_mag[i];
            td = store_clust[i];
            te = store_committor[i];
            k = i;
            while(i != (j = p_store_mag[k]-store_mag)){
                store_ngrid[k] = store_ngrid[j];
                store_slice[k] = store_slice[j];
                store_mag[k] = store_mag[j];
                store_clust[k] = store_clust[j];
                store_committor[k] = store_committor[j];
                p_store_mag[k] = &store_mag[k];
                k = j;
            }
            store_ngrid[k] = ta;
            store_slice[k] = tb;
            store_mag[k] = tc;
            store_clust[k] = td;
            store_committor[k] = te;
            p_store_mag[k] = &store_mag[k];
        }
    }

    // Create array for storing starting index of each cluster size and how many of a given cluster exist
    int out_num = store_mag[0];
    int unique_mag = 1;
    for (i=1;i<tot_samples;i++){
        mag = store_mag[i];
        if(mag != out_num){
            out_num = mag;
            unique_mag++;
        }
    }
    int *store_mag_index = (int *)malloc(unique_mag*2*sizeof(int));
    if (store_mag_index==NULL){fprintf(stderr,"Error allocating memory for store_mag_index array!\n"); exit(EXIT_FAILURE);}
    for (i=0;i<unique_mag*2;i++) {store_mag_index[i]=0;}

    int mag_check = store_mag[0];
    store_mag_index[0] = 0; // start index of magnetization
    store_mag_index[1] = 1; // how many instances of magnetization exist
    j = 0;

    for (i=1;i<tot_samples;i++) {
        mag = store_mag[i];
        if (mag > mag_check) {
            mag_check = mag;
            j++;
            store_mag_index[j*2] = i;
        }
        store_mag_index[j*2+1]++;
    }

    int random_selection = 0; // Used to randomly select grid from index.bin file
    int sample_tmp, sample_count;

    // Create array for storing random selections
    int *rand_array = (int *)malloc(samples*sizeof(int));
    if (rand_array==NULL){fprintf(stderr,"Error allocating memory for rand_array array!\n"); exit(EXIT_FAILURE);}
    int *loop_mag_index = (int *)malloc(unique_mag*sizeof(int));
    if (loop_mag_index==NULL){fprintf(stderr,"Error allocating memory for loop_mag_index array!\n"); exit(EXIT_FAILURE);}
    for (i=0;i<samples;i++) {rand_array[i]=-1;}
    for (i=0;i<unique_mag;i++) {loop_mag_index[i]=0;}

    int counter = 0;
    int in, im, rn, rm, im_tmp;
    int iterations = 0;

    sample_count = 0; j = 1;
    iterations = 1;
    while (sample_count <= samples) {
        for (i=0;i<unique_mag;i++) {
            if (store_mag_index[i*2+1] < iterations+1) {
                sample_count++;
            }
        }
        iterations++;
    }
    
    sample_count = 0;
    sample_tmp = 0;
    for (i=0;i<iterations;i++) {
        sample_tmp = 0;
        for (j=0;j<unique_mag;j++) {
            if (store_mag_index[j*2+1] > i) {
                loop_mag_index[sample_tmp] = j;
                sample_tmp++;
            }
        }
        im_tmp = sample_tmp;
        if (samples < sample_tmp) {
            im_tmp = samples;
        }
        im = 0;
        for (in = 0; in < sample_tmp && im < im_tmp; ++in) {
            rn = sample_tmp - in;
            rm = im_tmp - im;
            if (rand() % rn < rm) {
                im++;
                if (sample_count == samples) {
                    break;
                }
                rand_array[sample_count++] = loop_mag_index[in];
            }
        }
        if (sample_count == samples) {
            break;
        }
    }

    int **p_rand_array = malloc(samples*sizeof(long));

    // create array of pointers to store_cluster
    for (i = 0; i < samples; i++) {
        p_rand_array[i] = &rand_array[i];
    }

    // sort array of pointers
    qsort(p_rand_array, samples, sizeof(long), compare);

    // reorder loaded arrays according to the array of pointers
    for(i=0;i<samples;i++){
        if(i != p_rand_array[i]-rand_array){
            ta = rand_array[i];
            k = i;
            while(i != (j = p_rand_array[k]-rand_array)){
                rand_array[k] = rand_array[j];
                p_rand_array[k] = &rand_array[k];
                k = j;
            }
            rand_array[k] = ta; 
            p_rand_array[k] = &rand_array[k];
        }
    }

    i = 0; j = 0; out_num = -1;
    int rand_array_start = 0;
    while(i < samples){
        if(rand_array[i] != out_num && rand_array[i] > -1){
            out_num = rand_array[i];
            j += 1;
        }
        if(rand_array[i] < 0) {
            rand_array_start++;
        }
        i++;
    }

    int unique_rand = j;
    int *unique_rand_array = (int *)malloc(unique_rand*2*sizeof(int));
    if (unique_rand_array==NULL){fprintf(stderr,"Error allocating memory for unique_rand_array array!\n"); exit(EXIT_FAILURE);}
    for (i = 0; i < unique_rand*2; i++) {unique_rand_array[i] = -1;}

    i = rand_array_start+1; j = 1; k = 1;
    unique_rand_array[0] = rand_array[rand_array_start];
    while(i<samples){
        if(rand_array[i] != unique_rand_array[(j-1)*2] && rand_array[i] > -1){
            unique_rand_array[j*2] = rand_array[i];
            unique_rand_array[(j-1)*2+1] = k;
            k = 0;
            j++;
        }
        i++; k++;
    }
    unique_rand_array[(j-1)*2+1] = k;

    int *rand_array_sub = (int *)malloc((iterations+1)*sizeof(int));
    if (rand_array_sub==NULL){fprintf(stderr,"Error allocating memory for rand_array_sub array!\n"); exit(EXIT_FAILURE);}
    for (i = 0; i < iterations+1; i++) {rand_array_sub[i] = 0;}

    for (i = 0; i < unique_rand; i++) {
        im = 0;
        for (in = 0; in < store_mag_index[unique_rand_array[i*2]*2+1] && im < unique_rand_array[i*2+1]; ++in) {
            rn = store_mag_index[unique_rand_array[i*2]*2+1] - in;
            rm = unique_rand_array[i*2+1] - im;
            if (rand() % rn < rm) {
                rand_array_sub[im++] = in;               
            }
        }
        for (j = 0; j < unique_rand_array[i*2+1]; j++) {
            random_selection = rand_array_sub[j]+store_mag_index[unique_rand_array[i*2]*2];
            fwrite(&store_slice[random_selection], sizeof(int), 1, committor_file);
            fwrite(&store_ngrid[random_selection], sizeof(int), 1, committor_file);
            fwrite(&store_mag[random_selection], sizeof(int), 1, committor_file);
            fwrite(&store_clust[random_selection], sizeof(int), 1, committor_file);
            fwrite(&store_committor[random_selection], sizeof(double), 1, committor_file);
            fwrite(&store_committor[random_selection], sizeof(double), 1, committor_file); // Write to create space for standard deviation
            counter += 1;
        }
        printf("\rPercentage of available samples selected: %d%%", (int)(100.0*(double)counter/(double)(tot_samples))); // Print progress
        fflush(stdout);
    } 

    printf("\n"); // Newline

    free(store_ngrid); free(store_slice); free(store_mag); free(store_clust); free(store_committor); 
    free(store_ngrid_tmp); free(store_slice_tmp); free(store_mag_tmp); free(store_clust_tmp); free(store_committor_tmp); 
    free(rand_array); free(rand_array_sub); free(unique_rand_array);
    free(store_mag_index); free(loop_mag_index);
    free(p_store_mag); free(p_rand_array);

    fclose(index_file); fclose(committor_file);

    // Print time taken for program to execute
    end = clock();
    execution_time = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken: %.2f seconds \n", execution_time);
    
    return EXIT_SUCCESS;
}  
