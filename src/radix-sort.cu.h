#ifndef RADIX_SORT
#define RADIX_SORT

#include "kernel_env.cu.h"
#include "types.cu.h"

void compute_histogram(kernel_env env, uint32_t iteration){
    compute_histogram
        <<<env->num_blocks, env->block_size, 
            env->number_classes * sizeof(uint32_t)>>>(env->d_keys, env->d_hist, 
                                      env->bits, env->elem_pthread, 
                                      env->d_keys_size, env->number_classes, 
                                      iteration) ;
}

void transpose_histogram(kernel_env env){
    transpose<<<env->num_blocks, env->block_size>>>(env->d_hist_transpose, env->d_hist);
}

void scan_transposed_histogram(kernel_env env){
    // Determine temporary device storage requirements for inclusive prefix sum | CUB CODE
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, env->d_hist_transpose, env->d_hist_transpose, env->d_hist_size);
    // Allocate temporary storage for inclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, env->d_hist_transpose, env->d_hist_transpose, env->d_hist_size);
}

void get_condensed_scan(kernel_env env) {
    array_from_scan<<<1, env->number_classes>>>(env->scan_res, env->d_hist_transpose, env->d_hist_size, env->bits);
}

void scatter(kernel_env env, int iter) {
    scatter<<<env->num_blocks, env->block_size>>>(env->d_keys,
            env->d_output, env->scan_res, env->bits, env->elem_pthread, env->d_keys_size, 
            env->number_classes, iter);
}






#endif //!RADIX_SORT
