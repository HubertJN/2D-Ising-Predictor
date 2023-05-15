#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include "functions/functions.h" // Includes all function definitions

#define MOD(a,b) ((((a)%(b))+(b))%(b))

// Main function to find cluster size and write it to file

int main() {

    // Define and read input variables
    int L, nreplicas, nsweeps, mag_output_int, grid_output_int, threadsPerBlock, gpu_device, gpu_method;
    double beta, h;
    read_input_variables(&L, &nreplicas, &nsweeps, &mag_output_int, &grid_output_int, &threadsPerBlock, &gpu_device, &gpu_method, &beta, &h);

    // Define maximum connetions per grid point. 4 in this case since 2D nearest neighbour Ising model is being used
    int Maxcon = 4;
    int Nvert=L*L;

    // Set filenames
    const char *read_file_name = "gridstates.bin";
    const char *write_file_name = "index.bin";
    
    // open file to read grid
    FILE *read_file = fopen(read_file_name, "rb");
    if (read_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", read_file_name);
        exit(EXIT_FAILURE);
    }

    // Delete previous index file
    remove("index.bin");

    // create file to write indexing
    FILE *write_file = fopen(write_file_name, "wb");
    if (write_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", write_file_name);
        exit(EXIT_FAILURE);
    }

    // Host copy of Ising grid configurations
    int *ising_grids = (int *)malloc(L*L*sizeof(int));
    if (ising_grids==NULL){
        fprintf(stderr,"Error allocating memory for Ising grids!\n");
        exit(EXIT_FAILURE);
    } 

    // Define loop indices
    int islice,igrid;

    // Create output array for ngrid, islice, cluster and spare for commitor
    double *output = (double *)malloc(nreplicas*4*sizeof(double));
    if (output==NULL){
        fprintf(stderr,"Error allocating memory for output array!\n");
        exit(EXIT_FAILURE);
    } 

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer for ising grid!");
        exit(EXIT_FAILURE);
    }

    /*--------------------------------------------/
    / Allocate memory to hold graph connectivity  /
    /--------------------------------------------*/
    int temp_cluster_size = 0; int temp_number_of_clusters = 0;
    int *Ncon = (int *)malloc(Nvert*sizeof(int));
    if (Ncon==NULL) { printf("Error allocating Ncon array\n") ; exit(EXIT_FAILURE); }

    int *Lcon = (int *)malloc(Nvert*Maxcon*sizeof(int));
    if (Lcon==NULL) { printf("Error allocating Lcon array\n") ; exit(EXIT_FAILURE); }
    int i,j,k;
    j=0;k=0;

    // Main loop which finds cluster size and writes it to file
    // Loops over slices i.e. sweep snapshots

    for (islice=0;islice<nsweeps/100;islice++) {
        printf("\rNumber of sweeps processed: %d/%d", islice+1, nsweeps/100); // Print progress
        fflush(stdout);
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            read_input_grid(read_file, bitgrid, L, ising_grids, nreplicas, islice, igrid);
            //if (islice==0) {
            //for (i=0;i<L*L;i++) {printf("%d ", ising_grids[i]);}}
            // Saves grid number, slice, cluster size and spare data entry for commitor
            temp_cluster_size = find_cluster_size(L, Maxcon, ising_grids, Lcon, Ncon);
            output[igrid*4] = (double)igrid;
            output[igrid*4+1] = (double)islice*100;
            output[igrid*4+2] = (double)temp_cluster_size;
            output[igrid*4+3] = (double)-1;
        } // igrid
        fwrite(output, sizeof(double), nreplicas*4, write_file);
        
    } // isweep
/*
    read_input_grid(read_file, bitgrid, L, ising_grids, nreplicas, 10, 0);
    for (islice=0;islice<Nvert;islice++) {if (islice%L==0) {printf("\n");} if(ising_grids[islice]==1) {printf("1");} if(ising_grids[islice]==-1) {printf("0");} }
    k = find_cluster_size(L, Maxcon, ising_grids, Lcon, Ncon);
    printf("\n%d\n", k);
*/
    // Free memory
    free(bitgrid); free(output); free(ising_grids); free(Lcon); free(Ncon);

    // Close files
    fclose(write_file); fclose(read_file);
    
    // New line
    printf("\n");

    return EXIT_SUCCESS;
}


/*
 *
 * rowcol(i)
 *  row = i/L
 *  col = i%L
 *
 * index(row,col)
 *  index = col + row*L
*/
