#include <stdio.h>
#include <stdlib.h>
#include "functions.h" // Includes all function definitions

#define MOD(a,b) ((((a)%(b))+(b))%(b))

int find_cluster_size(int L, int Maxcon, int *grid, int* Lcon, int* Ncon) {

    /* Size of largest cluster, and number of clusters */
    int lclus,nclus,avlclus;
    /* Define indices */
    int i=0; int j=0;
    /* Number of vertices */
    int Nvert=L*L;
    /*----------------------/
    / Initialise averages   /
    /----------------------*/
    lclus = 0;
    nclus = 0;
    avlclus = 0;

    for (i=0;i<Nvert;i++)        { Ncon[i] =  0; } /* Initialise num. connections */
    for (i=0;i<Nvert*Maxcon;i++) { Lcon[i] = -1; } /* Initialise connection list  */

    //printf("Begin Ncon loop\n");
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
        //printf("Begin recursive \n");
        lclus = find_clusters_recursive(Nvert, Maxcon, Ncon, Lcon);
        //find_clusters_eq(Nvert,Maxcon,Ncon,Lcon,&lclus,&nclus);
        avlclus = lclus; 
    }
    else {
        avlclus = 0;
    }
    return avlclus;
}

// Function to find the largest cluster size and number of clusters
int find_clusters_recursive(int Nvert, int Maxcon, int *Ncon, int *Lcon) {

    int *lvisited = (int *)malloc(Nvert*sizeof(int)); // Logical array indicating if vertex has been counted
    int *cluster_size = (int *)malloc(Nvert*sizeof(int)); // Array holding size of each cluster

    int iv, ivcluster; // Loop counters
    int lmax = 0; // Maximum cluster counter

    // Initialise loop counters and error flags
    iv = 0; ivcluster = 0;

    // Initialise arrays
    for (iv=0;iv<Nvert;iv++) {lvisited[iv] = 0;}
    for (iv=0;iv<Nvert;iv++) {cluster_size[iv] = 0;}

    // Lopp over vertices
    //printf("Begin cluster size loop \n");
    for (iv=0;iv<Nvert;iv++) {
        if (lvisited[iv]==0) {
            lvisited[iv] = 1;

            // New cluster at current ivcluster
            cluster_size[ivcluster] = 0;

            // Add vertex iv to current cluster
            cluster_size[ivcluster] += 1;

            // Search onward over all edges involving this vertex
            // After function finish should have followed all possible links originating on vertex iv
            // all members of the cluster containing iv.
            //printf("Begin vertex search\n");
            vertex_search(iv, ivcluster, Maxcon, Ncon, Lcon, lvisited, cluster_size);

            // Next cluster
            ivcluster += 1;
        }
    }

    // Analysis complete. Output number of cluster and size of cluster
    // Compare elements of array with max
    for (iv=0;iv<Nvert;iv++) {if(cluster_size[iv]>lmax) {lmax=cluster_size[iv];}}
    //for (iv=0;iv<Nvert;iv++) {printf("%d ", cluster_size[iv]);}

    // Free memory
    free(lvisited); free(cluster_size);
    //printf("\n%d\n", lmax);
    return lmax;
}

// Recursive function for searching over the vertices
void vertex_search(int i, int icluster, int Maxcon, int *Ncon, int *Lcon, int *lvisited, int *cluster_size) {

    int jlist = 0;
    int j = 0;

    if (Ncon[i]==0) {return;}

    for (jlist=0;jlist<Ncon[i];jlist++) {
        j = Lcon[i*Maxcon+jlist];
        if (lvisited[j]==0) {

            lvisited[j] = 1;
            // Increment the size of icluster, and add j to the list of vertices in icluster
            cluster_size[icluster] += 1;
            // Search onward from here, calling this function recursively
            //if (icluster==0) {printf("%d ",cluster_size[icluster]);}
            vertex_search(j, icluster, Maxcon, Ncon, Lcon, lvisited, cluster_size);
        }
    }
}



