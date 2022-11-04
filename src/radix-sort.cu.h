#ifndef RADIX_SORT
#define RADIX_SORT
#include "cub/cub.cuh"

#include "kernels.cu.h"
#include "types.cu.h"
#include "kernel_env.cu.h"
#include "helper.cu.h"
#include "utils.cu.h"


void compute_histogram(kernel_env env, uint32_t iteration){
    compute_histogram
        <<<env->num_blocks, env->block_size, 
            env->number_classes * sizeof(uint32_t)>>>(env->d_keys, env->d_hist, 
                                      env->bits, env->elem_pthread, 
                                      env->d_keys_size, env->number_classes, 
                                      iteration);
}

void compute_histogram_local(kernel_env env, uint32_t iteration){
    debug("--- Calling histogram local kernel");
    size_t shared_memory_size = 
        env->block_size * 
        env->number_classes * 
        env->elem_pthread *  sizeof(uint16_t);
    fprintf(stderr, "shared_memeory_size: %d\n", shared_memory_size);
        
    compute_histogram_sort<<<env->num_blocks, env->block_size, 
        shared_memory_size>>>(env->d_keys, env->d_hist, 
                                env->bits, env->elem_pthread, 
                                env->d_keys_size, env->number_classes, 
                                iteration);

    cudaCheckError();
}

void transpose_histogram(kernel_env env){
    transpose<<<env->num_blocks, env->block_size>>>(env->d_hist_transpose, env->d_hist);
}

void example() {
    int block_size = 3;
    int elem_pthread = 4;
    int size = block_size * elem_pthread;
    int array[size];
    //for(int i = 0; i < size; ++i){
    //    if(i > 0 && i % 16 == 0) printf("\n");
    //    array[i] = -99;
    //    printf("%d, ", array[i]);
    //}
    //printf("\n");

    int *d_array;
    cudaMalloc((void**) &d_array,  size * sizeof(int));

    exampleKernel<<<1, block_size>>>(array);

    cudaMemcpy(array, d_array, size * sizeof(uint32_t), 
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < size; ++i){
        if(i > 0 && i % 16 == 0) printf("\n");
        printf("%d, ", array[i]);
    }

    printf("\n");
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

void scan_transposed_histogram_exclusive(kernel_env env){
    // Determine temporary device storage requirements for inclusive prefix sum | CUB CODE
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, env->d_hist_transpose, env->d_hist_transpose, env->d_hist_size);
    // Allocate temporary storage for exclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, env->d_hist_transpose, env->d_hist_transpose, env->d_hist_size);
}

void get_condensed_scan(kernel_env env) {
    array_from_scan<<<1, env->number_classes>>>(env->d_scan_res, env->d_hist_transpose, env->d_hist_size, env->bits);
}

void scatter(kernel_env env, int iter) {
    scatter<<<env->num_blocks, env->block_size>>>(env->d_keys,
            env->d_output, env->d_scan_res, env->bits, env->elem_pthread, env->d_keys_size, 
            env->number_classes, iter);
}

void radix_sort(kernel_env env){
    for(uint32_t it = 0; it < size_in_bits<uint32_t>(); it += env->bits){
        uint32_t iteration = it / env->bits;
        compute_histogram(env, iteration);
        transpose_histogram(env);
        scan_transposed_histogram(env);
        get_condensed_scan(env);
        scatter(env, iteration);
    }
}






#endif //!RADIX_SORT
