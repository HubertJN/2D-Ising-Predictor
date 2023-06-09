#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "input_reader.h"

// This considers the case of single spin blocks in other models d_Pacc may need to be larger
// Cache of acceptance probabilities 
__constant__ float d_Pacc[20];   // gpu constant memory

#endif // HELPERS_H