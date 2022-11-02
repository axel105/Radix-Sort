#ifndef TEST_KERNELS
#define TEST_KERNELS
#include "cub/cub.cuh"

#include "utils.cu.h"
#include "kernel_env.cu.h"
#include "types.cu.h"
#include "radix-sort-kernels.cu.h"
#include "radix-sort.cu.h"
#include "radix-sort-cpu.cu.h"

bool equal(uint32_t* vec1, uint32_t* vec2, const uint32_t size){
    for(uint32_t i = 0; i < size; ++i){
        if(vec1[i] != vec2[i]) return false;
    }
    return true;
}

bool test_compute_histogram(kernel_env env){
    fprintf(stderr, "*** Testing compute histogram kernel!\n");
    // compute histogram on CPU
    compute_histogram(env->h_keys, env->h_hist, env->bits, 
                      env->h_keys_size, 0);

    // compute histogram on GPU
    compute_histogram(env, 0);

    fprintf(stderr, "\n-- Expected histogram (CPU):\n");
    log_vector(env->h_hist, env->h_hist_size);
    fprintf(stderr, "\n-- Result histogram (GPU):\n");
    log_reduce_d_hist(env);

    fprintf(stderr, "\n-- Unreduced histogram (GPU):\n");
    if(DEBUG) log_d_hist(env);

    uint32_t hist_size = d_hist_size(env), histogram[hist_size];
    reduce_d_hist(env, histogram);

    return equal(env->h_hist, histogram, env->h_hist_size);
}

bool test_transpose(kernel_env env){
    fprintf(stderr, "*** Testing histogram tranposition kernel!\n");
    transpose_histogram(env);
    log_d_hist_transpose(env);
    return false;
}

bool test_scan(kernel_env env){
    fprintf(stderr, "*** Testing scan on transposed histogram!\n");
    scan_transposed_histogram(env);
    log_d_hist_transpose(env);
    return false;
    
}



#endif //!TEST_KERNELS

