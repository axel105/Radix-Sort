#ifndef UTILS
#define UTILS

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>


/**
  * Initialize an array with random values withing the range of H.
  *
  * @data: array to write values to
  * @size: size of @data
  * @H: maximum bounds for random numbers
  */
void randomInitNat(int* data, const int size, const int H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

#endif // !UTILS
