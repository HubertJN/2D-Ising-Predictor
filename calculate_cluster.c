#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

#define MOD(a,b) ((((a)%(b))+(b))%(b))

/* Function prototypes for cluster analysis routines */
void find_clusters_recursive  (long Nvert,long Maxcon,long *Ncon,long *Lcon,long *lclus, long *nclus);
void find_clusters_eqclass  (long Nvert,long Maxcon,long *Ncon,long *Lcon,long *lclus, long *nclus);
long find_cluster_size(int L,int Maxcon,int *grid);
void read_input_grid(FILE *file, char *bitgrid, int L, int *ising_grids, int grids_per_slice, int islice, int igrid);
void write_cluster(int *cluster, FILE *ptr, int grids_per_slice);

// Main function to find cluster size and write it to file

int main() {

    // Define input variables
    int L = 64;
    int grids_per_slice = 1;
    int Maxcon = 4;
    int tot_sweeps = 5000;

    // Set filenames
    const char *filename1 = "index.bin";
    const char *filename2 = "gridstates.bin";

    // Remove old output
    remove("index.bin");
    
    // open write cluster file
    FILE *ptr1 = fopen(filename1,"ab");
    if (ptr1==NULL){
        FILE *ptr1 = fopen(filename1,"wb"); // open for write if not available for append 
        if (ptr1==NULL){
            fprintf(stderr,"Error opening %s for write!\n",filename1);
            exit(EXIT_FAILURE);
        }
    }
    
    // open read grid file
    FILE *ptr2 = fopen(filename2, "rb");
    if (ptr2==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename2);
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
    int *output = (int *)malloc(grids_per_slice*3*sizeof(int));
    if (output==NULL){
        fprintf(stderr,"Error allocating memory for Ising grids!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer!");
        exit(EXIT_FAILURE);
    }

    // Main loop which finds cluster size and writes it to file
    // Loops over slices i.e. sweep snapshots
    for (islice=0;islice<tot_sweeps/100;islice++) {
        //printf("\rNumber of sweeps processed: %d/%d", islice+1,tot_sweeps/100); // Print progress
        //fflush(stdout);
        // Loops over grids of each sweep snapshot  
        for (igrid=0;igrid<grids_per_slice;igrid++) {
            read_input_grid(ptr2, bitgrid, L, ising_grids, grids_per_slice, islice, igrid);
            // Saves sweep, grid and cluster size to output array
            output[igrid*3] = islice*100;
            output[igrid*3+1] = igrid;
            output[igrid*3+2] = find_cluster_size(L,Maxcon,ising_grids);
            printf("%d %d %d\n", output[igrid], output[igrid+1], output[igrid+2]);
        } // igrid
        write_cluster(output, ptr1, grids_per_slice);
    } // isweep

    // Free memory
    free(bitgrid); free(output); free(ising_grids);

    // Close files
    fclose(ptr2); fclose(ptr1);
    
    // New line
    printf("\n");

    return EXIT_SUCCESS;
}

long find_cluster_size(int L,int Maxcon,int *grid) {

    /* Size of largest cluster, and number of clusters */
    long lclus,nclus,avlclus,avnclus;
    /* Define indices */
    int i=0; int j=0;
    /* Number of vertices */
    int Nvert=L*L;
    /* Pointers for connections and number of edges */
    long *Lcon;
    long *Ncon;
    /*----------------------/
    / Initialise averages   /
    /----------------------*/
    avlclus = 0.0;
    avnclus = 0.0;

    /*--------------------------------------------/
    / Allocate memory to hold graph connectivity  /
    /--------------------------------------------*/

    Ncon = (long *)malloc(Nvert*sizeof(long));
    if (Ncon==NULL) { printf("Error allocating Ncon array\n") ; exit(EXIT_FAILURE); }

    Lcon = (long *)malloc(Nvert*Maxcon*sizeof(long));
    if (Lcon==NULL) { printf("Error allocating Lcon array\n") ; exit(EXIT_FAILURE); }

    for (i=0;i<Nvert;i++)        { Ncon[i] =  0; } /* Initialise num. connections */
    for (i=0;i<Nvert*Maxcon;i++) { Lcon[i] = -1; } /* Initialise connection list  */

    for (i=0; i<L*L; i++) {
        if (grid[i] == 1) {
            j = MOD(i+1,L) + L*(i/L); // Find index of adjacent vertex
            if (grid[j] == 1) { // Check neighbour is also equal to 1
                Ncon[i] += 1; // Increment edges involving i
                Lcon[Maxcon*i+Ncon[i]-1] = j; // j is connected to i 
            }
            j = MOD(i-1,L) + L*(i/L);
            if (grid[j] == 1) {
                Ncon[i] += 1;
                Lcon[Maxcon*i+Ncon[i]-1] = j;
            }
            j = MOD(i+L,L*L);
            if (grid[j] == 1){
                Ncon[i] += 1;
                Lcon[Maxcon*i+Ncon[i]-1] = j;
            }
            j = MOD(i-L,L*L);
            if (grid[j] == 1) {
                Ncon[i] += 1;
                Lcon[Maxcon*i+Ncon[i]-1] = j;
            }

            /* Check that we will not overrun the end of the Lcon array */
            if ( ( Ncon[i] > Maxcon ) ) {
                printf("Maximum number of edges per vertex exceeded!\n");
                exit(EXIT_FAILURE);
            }
        } /* if */
    } /* i */

    int con_exist = 0;
    for (i=0;i<Nvert;i++) { 
        if (Ncon[i] !=  0) {
            con_exist = 1;
            break;
        } 
    }

    if (con_exist == 1) {
        find_clusters_eqclass(Nvert,Maxcon,Ncon,Lcon,&lclus,&nclus);
        avlclus = (double)lclus; 
    }
    else {
        avlclus = 0;
    }

    /*----------------/
    / Release memory  /
    /----------------*/
    free(Lcon);
    free(Ncon);
    
    return (long)avlclus;
}

void read_input_grid(FILE *ptr, char *bitgrid, int L, int *ising_grids, int grids_per_slice, int islice, int igrid){
    
    // bytes per slice to move through gridfile
    int bytes_per_slice = 12+grids_per_slice*(L*L/8);

    // converts [0,1] to [-1,1]
    const int blookup[2] = {-1, 1};

    uint32_t one = 1U;

    int nbytes = L*L/8;
    // Read the grid
    fseek(ptr, 12+bytes_per_slice*islice+(L*L/8)*(igrid), SEEK_SET);
    fread(bitgrid, sizeof(char), nbytes, ptr);

    // Loop over grid points
    int ibit=0, ibyte=0;
    int isite=0;
    for (ibyte=0;ibyte<nbytes;ibyte++){
        for (ibit=0;ibit<8;ibit++){
            ising_grids[isite] = blookup[(bitgrid[ibyte] >> ibit) & one];
            isite++;
            if (isite>L*L) {break;}
        }
    }
}

void write_cluster(int *cluster, FILE *ptr, int grids_per_slice){
    fwrite(cluster,sizeof(int),grids_per_slice*3,ptr);
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