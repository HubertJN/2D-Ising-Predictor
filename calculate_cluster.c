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


    // Set filenames
    const char *write_file_name = "index.bin";
    const char *read_file_name = "gridstates.bin";

    // Remove old output
    remove("index.bin");
    
    // create write cluster file
    FILE *write_file = fopen(write_file_name,"wb");
    if (write_file==NULL){
        fprintf(stderr,"Error creating %s for write!\n",write_file_name);
        exit(EXIT_FAILURE);
    }
    
    
    // open read grid file
    FILE *read_file = fopen(read_file_name, "rb");
    if (read_file==NULL){
        fprintf(stderr, "Error opening %s for input!\n", read_file_name);
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

    // Create output array to contain [sweep, grid, cluster size]
    int *output = (int *)malloc(nreplicas*3*sizeof(int));
    if (output==NULL){
        fprintf(stderr,"Error allocating memory for output write array!\n");
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
    int Nvert=L*L;
    int *Ncon = (int *)malloc(Nvert*sizeof(int));
    if (Ncon==NULL) { printf("Error allocating Ncon array\n") ; exit(EXIT_FAILURE); }

    int *Lcon = (int *)malloc(Nvert*Maxcon*sizeof(int));
    if (Lcon==NULL) { printf("Error allocating Lcon array\n") ; exit(EXIT_FAILURE); }

    // Main loop which finds cluster size and writes it to file
    // Loops over slices i.e. sweep snapshots
    for (islice=0;islice<nsweeps/100;islice++) {
        printf("\rNumber of sweeps processed: %d/%d", islice+1, nsweeps/100); // Print progress
        fflush(stdout);
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<nreplicas;igrid++) {
            read_input_grid(read_file, bitgrid, L, ising_grids, nreplicas, islice, igrid);
            // Saves sweep, grid and cluster size to output array
            output[igrid*3] = islice*100;
            output[igrid*3+1] = igrid;
            output[igrid*3+2] = find_cluster_size(L, Maxcon, ising_grids, Lcon, Ncon);
            //printf("%d %d %d\n", output[igrid*3], output[igrid*3+1], output[igrid*3+2]);
        } // igrid
        fwrite(output, sizeof(int), nreplicas*3, write_file);
    } // isweep
    
    // Free memory
    free(bitgrid); free(output); free(ising_grids); free(Lcon); free(Ncon);

    // Close files
    fclose(read_file); fclose(write_file);
    
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
