#ifndef UTILS
#define UTILS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include "types.cu.h"

#define BYTES_IN_BITS 8

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

// Parse the input arguments
int parse_args(int argc, char **argv){
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    return atoi(argv[1]);
}

void log_vector(uint32_t *vec, uint32_t size){
    fprintf(stderr, "[");
    for(uint32_t i = 0; i < size; ++i){
        fprintf(stderr, " %d,", vec[i]);
    }
    fprintf(stderr, " ]\n");
}

void debug(const char *msg){
    if(DEBUG) fprintf(stderr, "%s\n", msg);
}

void log_vector_with_break(uint32_t *vec, uint32_t size, uint32_t line_break){
    fprintf(stderr, "[");
    for(uint32_t i = 0; i < size; ++i){
        if (i > 0 && i % line_break == 0) fprintf(stderr, "\n");
        fprintf(stderr, " %d,", vec[i]);
    }
    fprintf(stderr, " ]\n");
}

template <typename T>
size_t size_in_bits(){
    return sizeof(T) * BYTES_IN_BITS;
}


#endif // !UTILS
