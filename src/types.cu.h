#ifndef TYPE
#define TYPE
#define DEBUG 1

typedef unsigned int uint32_t; 

typedef struct Kernel_env{
    // general data attributes
    uint32_t number_classes;
    uint32_t h_keys_size;
    uint32_t h_hist_size;
    uint32_t max_value;
    uint32_t *h_keys;
    uint32_t *h_hist;

    // GPU data attributes
    uint32_t* d_keys;
    uint32_t* d_output;
    uint32_t* d_hist;
    uint32_t* d_hist_transpose;
    uint32_t d_keys_size;
    uint32_t d_hist_size;

    // GPU settings
    uint32_t num_blocks;
    uint32_t block_size;
    uint32_t elem_pthread;
    uint32_t bits;
}*kernel_env;


#endif //TYPE
