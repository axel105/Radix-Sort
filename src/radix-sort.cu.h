#ifndef RADIX_SORT
#define RADIX_SORT
#include "cub/cub.cuh"

#include "helper.cu.h"
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "types.cu.h"
#include "utils.cu.h"


void compute_histogram_local(kernel_env env, uint32_t iteration) {
    size_t shared_memory_size = env->block_size * env->number_classes *
                                env->elem_pthread * sizeof(uint16_t);

    compute_histogram_sort<<<env->num_blocks, env->block_size,
                             shared_memory_size>>>(
        env->d_keys, env->d_hist, env->d_hist_scan, env->bits,
        env->elem_pthread, env->d_keys_size, env->number_classes, iteration);

    cudaCheckError();
}

void transpose_histogram(kernel_env env) {
    transpose<<<env->num_blocks, env->block_size>>>(env->d_hist_transpose,
                                                    env->d_hist);
}


void scan_transposed_histogram(kernel_env env) {
    // Determine temporary device storage requirements for inclusive prefix sum
    // | CUB CODE
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  env->d_hist_transpose, env->d_hist_transpose,
                                  env->d_hist_size);
    // Allocate temporary storage for inclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  env->d_hist_transpose, env->d_hist_transpose,
                                  env->d_hist_size);
}

void scan_transposed_histogram_exclusive(kernel_env env) {
    // Determine temporary device storage requirements for inclusive prefix sum
    // | CUB CODE
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  env->d_hist_transpose, env->d_hist_transpose,
                                  env->d_hist_size);
    // Allocate temporary storage for exclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  env->d_hist_transpose, env->d_hist_transpose,
                                  env->d_hist_size);
}

void scatter_coalesced(kernel_env env, int iter) {
    scatter_coalesced<<<env->num_blocks, env->block_size>>>(
        env->d_keys, env->d_output, env->d_hist_transpose, env->d_hist,
        env->d_hist_scan, env->bits, env->elem_pthread, env->d_keys_size,
        env->d_hist_size, iter);
}

void radix_sort(kernel_env env) {
    // for (uint32_t it = 0; it < size_in_bits<uint32_t>(); it += env->bits) {
    // uint32_t iteration = it / env->bits;
    for (uint32_t iteration = 0; iteration < size_in_bits<uint32_t>(); iteration++) {
        //fprintf(stderr, "****** ITERATION: %d\n", iteration);
        compute_histogram_local(env, iteration);
        //debug("---- Compute histogram");
        //debug("Input array");
        //log_d_keys(env);
        cudaDeviceSynchronize();
        transpose_histogram(env);
        cudaDeviceSynchronize();
        scan_transposed_histogram_exclusive(env);
        cudaDeviceSynchronize();
        scatter_coalesced(env, iteration);
        //debug("---- Scatter (before swap)");
        //debug("Input array");
        //log_d_keys(env);
        //debug("Output array");
        //log_output_result(env);
        uint32_t *tmp = env->d_keys;
        env->d_keys = env->d_output;
        env->d_output = tmp;
        cudaDeviceSynchronize();
    }
}

#endif  //! RADIX_SORT
