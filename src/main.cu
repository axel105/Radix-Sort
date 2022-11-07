#include <stdio.h>
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "radix-sort.cu.h"
#include "types.cu.h"
#include "utils.cu.h"

double sortRedByKeyCUB( uint32_t* data_keys_in
                      , uint32_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);

    return elapsed;
}

double bench_radix_sort(kernel_env env){
    { // one dry run
        radix_sort(env);
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        radix_sort(env);
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    return elapsed;
}


int main(int argc, char **argv) {
    const int number_keys = parse_args(argc, argv);

    const uint32_t block_size = 256, elem_pthread = 4, bits = 4, max_value = 16;
    kernel_env env = 
        new_kernel_env(block_size, elem_pthread, bits, number_keys, max_value);
    kernel_env cub_env = copy(env);

    //Allocate and Initialize Device data
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  number_keys * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, env->h_keys, number_keys * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, number_keys * sizeof(uint32_t)));


    double elapsed = bench_radix_sort(env);
    printf("Radix Sorting for N=%lu runs in: %.2f us\n", number_keys, elapsed);

    double cub_elapsed = sortRedByKeyCUB(d_keys_in, d_keys_out, number_keys);
    printf("CUB Sorting for N=%lu runs in: %.2f us\n", number_keys, cub_elapsed);

    free_env(env);
    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);

    return 0;
}
