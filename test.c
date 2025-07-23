#include <stdio.h>
#include <stdlib.h>

#define MOD(a,b) ((((a)%(b))+(b))%(b))

int main() {
    // Set filenames
    const char *filename1 = "committor_index.bin";

    // open write cluster file
    FILE *ptr1 = fopen(filename1,"rwb"); // open for write if not available for append 
    if (ptr1==NULL){
        fprintf(stderr,"Error opening %s for write!\n",filename1);
        exit(EXIT_FAILURE);
    }

    // Create array to store index
    int *index = (int *)malloc(4*sizeof(int));
    if (index==NULL){
        fprintf(stderr,"Error allocating memory for index!\n");
        exit(EXIT_FAILURE);
    }

    int i = 0;
    double tmp1 = 0.0, tmp2 = 0.0;

    while (1) {
        fread(index, sizeof(int), 4, ptr1);
        fread(&tmp1, sizeof(double), 1, ptr1);
        fread(&tmp2, sizeof(double), 1, ptr1);
        if ( feof(ptr1) ) { break;}
        //if (index[3] > 50) {
        //if (MOD(i, 100) == 33) {
            printf("%d %d %d %d %f %f \n", index[0], index[1], index[2], index[3], tmp1, tmp2);
        //}
        i++;
    }
    //printf("\n");
    return(EXIT_SUCCESS);
}
