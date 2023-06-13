#include "comparison.h"

// Comparison function for sorting
int compare(const void *p_to_p_0, const void *p_to_p_1) {
    int i = **(int **)p_to_p_0;
    int j = **(int **)p_to_p_1;
    if(i > j)return 1;
    if(i < j)return -1;
    return 0;
}