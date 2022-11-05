#ifndef RADIX_CPU
#define RADIX_CPU
#include "types.cu.h"

void compute_histogram(uint32_t *h_keys, uint32_t *hist, uint32_t bits,
                       uint32_t in_size, uint32_t it) {
    // 2**bits (give the number of classes)
    const uint32_t number_classes = 1 << bits;
    int histogram[number_classes];
    bzero(histogram, sizeof(int) * number_classes);

    // compute each keys
    for (uint32_t i = 0; i < in_size; ++i) {
        // get 4 bits to be at the beginning and determine the rank
        uint32_t rank = h_keys[i] >> (it * bits) & (number_classes - 1);
        ++histogram[rank];
    }
    for (uint32_t i = 0; i < number_classes; ++i) {
        hist[i] = histogram[i];
    }
}

#endif  // !RADIX_CPU
