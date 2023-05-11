#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "functions.h" // Includes all function definitions

void read_input_grid(FILE *ptr, char *bitgrid, int L, int *ising_grids, int grids_per_slice, int islice, int igrid) {
    
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