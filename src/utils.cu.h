#ifndef UTILS
#define UTILS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>

typedef unsigned int uint32_t; 


/**
  * Initialize an array with random values withing the range of H.
  *
  * @data: array to write values to
  * @size: size of @data
  * @H: maximum bounds for random numbers
  */
void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

void log_vec(char* name, uint32_t *vec, uint32_t size){
    fprintf(stderr, "%s: [", name);
    for(uint32_t i = 0; i < size; ++i){
        if (i % 16 == 0) fprintf(stderr, "\n");
        fprintf(stderr, " %d,", vec[i]);
    }
    fprintf(stderr, " ]\n");
    
}

#endif // !UTILS
