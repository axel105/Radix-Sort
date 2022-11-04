#include <stdio.h>

#include "test_utils.cu.h"
#include "cpu_functions.cu.h"

// from src
#include "radix-sort.cu.h"
#include "utils.cu.h"
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "types.cu.h"

bool test_compute_histogram(kernel_env env){
    fprintf(stderr, "*** Testing compute histogram kernel!\n");
    for(uint32_t it = 0; it < size_in_bits<uint32_t>(); it += env->bits){
        uint32_t iteration = it / env->bits;
        // compute histogram on CPU
        compute_histogram(env->h_keys, env->h_hist, env->bits, 
                          env->h_keys_size, iteration);

        // compute histogram on GPU
        compute_histogram(env, iteration);

        // Validate histograms
        uint32_t hist_size = d_hist_size(env), histogram[hist_size];
        reduce_d_hist(env, histogram);

        if(!equal(env->h_hist, histogram, env->number_classes)){
            if(DEBUG){
                fprintf(stderr, "Failure at iteration: %d\n", iteration);
                fprintf(stderr, "\n-- Expected histogram (CPU):\n");
                log_vector(env->h_hist, env->h_hist_size);
                fprintf(stderr, "\n-- Result histogram (GPU):\n");
                log_reduce_d_hist(env);
                fprintf(stderr, "\n-- Unreduced histogram (GPU):\n");
                log_d_hist(env);
            }
            return false;
        }
    }
    return true;
}

bool test_compute_histogram_local(kernel_env env){
    fprintf(stderr, "*** Testing compute histogram local kernel!\n");

    // compute histogram on GPU
    compute_histogram_local(env, 0);

    //log_d_hist(env);
    return false;

}

bool test_transpose(kernel_env env){
    fprintf(stderr, "*** Testing histogram tranposition kernel!\n");
    transpose_histogram(env);
    log_d_hist_transpose(env);
    return true;
}

bool test_scan(kernel_env env){
    fprintf(stderr, "*** Testing scan on transposed histogram!\n");
    scan_transposed_histogram(env);
    log_d_hist_transpose(env);
    return false;
}

bool test_get_scan_result(kernel_env env) {
    fprintf(stderr, "*** Get condensed scan result\n");
    get_condensed_scan(env);
    log_scan_result(env);
    return false;
}

bool test_scatter(kernel_env env) {
    fprintf(stderr, "*** Scattering\n");
    scatter(env, 0);
    log_output_result(env);
    return false;
}  

bool test_radix_sort(kernel_env env){
    radix_sort(env);
    log_d_keys(env);
    log_output_result(env);
    return false;
}

bool test(bool (*f)(kernel_env), kernel_env env){
    bool success = f(env);
    success ? 
        fprintf(stderr, "Test passed!\n") : fprintf(stderr, "Test FAILED!\n");
    return success;
}

int main(int argc, char **argv){
    bool success = true;
    const int number_keys = parse_args(argc, argv);

    const uint32_t block_size = 256, elem_pthread = 4, 
    bits = 4, max_value = 16;

    kernel_env env = new_kernel_env(block_size, elem_pthread,
                                    bits, number_keys, max_value);

    //success |= test(test_compute_histogram, env);
    test_compute_histogram_local(env);
    log_d_keys(env);
    //log_d_hist(env);
    //log_d_hist_scan(env);
    cudaDeviceSynchronize();
    cudaCheckError();

    free_env(env);

    return success ? 0 : 1;
}
