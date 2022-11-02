#ifndef RADIX_CPU
#define RADIX_CPU

void compute_histogram(uint32_t *h_keys, 
                        uint32_t *hist, uint32_t bits, 
                        uint32_t in_size, uint32_t it){
    // 2**bits (give the number of classes)
    const uint32_t number_classes = 1 << bits; 

    // compute each keys
    for(uint32_t i = 0; i < in_size; ++i){
        // get 4 bits to be at the beginning and determine the rank
        uint32_t rank = h_keys[i] >> (it * bits) & (number_classes -1);
        ++hist[rank];
    }
}

#endif // !RADIX_CPU
