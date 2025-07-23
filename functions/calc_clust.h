#include <stdio.h>
#include <stdlib.h>

int calculate_cluster(int L, int Maxcon, int *grid, int* Lcon, int* Ncon);
int calculate_clusters_recursive(int Nvert, int Maxcon, int *Ncon, int *Lcon);
void vertex_search(int i, int icluster, int Maxcon, int *Ncon, int *Lcon, int *lvisited, int *cluster_size);

