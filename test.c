#include <stdio.h>
#include <stdlib.h>

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
    int *index = (int *)malloc(3*sizeof(int));
    if (index==NULL){
        fprintf(stderr,"Error allocating memory for index!\n");
        exit(EXIT_FAILURE);
    }

    int tot_sample = 0;
    int within_range = 0;
    double tmp1 = 0.0, tmp2 = 0.0;
    double lower = 0.0, upper = 1.0;
    int min = 64*64, max = 0;

    while (1) {
        fread(index, sizeof(int), 3, ptr1);
        fread(&tmp1, sizeof(double), 1, ptr1);
        fread(&tmp2, sizeof(double), 1, ptr1);
        if ( feof(ptr1) ) { break;}
        printf("%d %d %d %f %f \n", index[0], index[1], index[2], tmp1, tmp2);
        if (tmp1 < upper && tmp1 > lower) {
            within_range += 1;
            if (index[2] > max) {
                max = index[2];
            }
            if (index[2] < min) {
                min = index[2];
            }
        }
        tot_sample += 1;
    }
    printf("Total Samples: %d \n", tot_sample);
    printf("Samples with Commitor between %.2f and %.2f: %d \n", lower, upper, within_range);
    printf("Minimum and Maximum cluster sizes within commitor range: %d %d \n", min, max);

    return(EXIT_SUCCESS);
}