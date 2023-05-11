#include <stdio.h>
#include <stdlib.h>

void read_input_grid(FILE *ptr, char *bitgrid, int L, int *ising_grids, int grids_per_slice, int islice, int igrid);
int find_cluster_size(int L, int Maxcon, int *grid, int* Lcon, int* Ncon);
void find_clusters_recursive(int Nvert, int Maxcon, int *Ncon, int *Lcon, int *lclus, int *nclus);
void vertex_search(int i, int icluster, int Maxcon, int *Ncon, int *Lcon, int *lvisited, int *cluster_size);
void read_input_variables(int *L, int *nreplicas, int *nsweeps, int *mag_output_int, int *grid_output_int, int *threadsPerBlock, int *gpu_device, int *gpu_method, double *beta, double *h);
