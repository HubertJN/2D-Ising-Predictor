#include <stdio.h>
#include <stdlib.h>

int find_cluster_size(int L, int Maxcon, int *grid, int* Lcon, int* Ncon);
int find_clusters_recursive(int Nvert, int Maxcon, int *Ncon, int *Lcon);
int find_clusters_eqclass(int Nvert, int Maxcon, int *Ncon, int *Lcon);
void vertex_search(int i, int icluster, int Maxcon, int *Ncon, int *Lcon, int *lvisited, int *cluster_size);

