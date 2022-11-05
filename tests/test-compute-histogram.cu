#include "cpu_functions.cu.h"
#include "test_utils.cu.h"

// from src
#include "helper.cu.h"
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "radix-sort.cu.h"
#include "types.cu.h"
#include "utils.cu.h"

#define NUM_BLOCKS_NEEDED 1
#define INPUT_MAX_SIZE 1024 * 16 * NUM_BLOCKS_NEEDED

bool valid_histogram(kernel_env env) {
    debug(
        "- Testing histogram compute a correct histogram, and scanned "
        "histogram...");
    for (uint32_t it = 0; it < size_in_bits<uint32_t>(); it += env->bits) {
        uint32_t iteration = it / env->bits;
        // compute histogram on CPU
        compute_histogram(env->h_keys, env->h_hist, env->bits, env->h_keys_size,
                          iteration);

        // compute histogram on GPU
        compute_histogram_local(env, iteration);
        uint32_t hist_size = d_hist_size(env), histogram[hist_size];
        reduce_d_hist(env, histogram);

        if (!equal(env->h_hist, histogram, env->number_classes)) {
            if (DEBUG) {
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

bool sorted_properties(kernel_env) {
    debug("- Testing that the keys are sorted at each steps...");

    // compute histogram on GPU
    compute_histogram_local(env, iteration);

    return false;
}

int main(int argc, char **argv) {
    bool success = true;
    // const int number_keys = parse_args(argc, argv);
    for (uint32_t number_keys = 1; number_keys < INPUT_MAX_SIZE;
         ++number_keys) {
        const uint32_t block_size = 256, elem_pthread = 4, bits = 4,
                       max_value = number_keys / 10;

        kernel_env env = new_kernel_env(block_size, elem_pthread, bits,
                                        number_keys, max_value);

        success |= test(valid_histogram, env);
        success |= test(sorted_properties, env);

        cudaDeviceSynchronize();
        cudaCheckError();

        free_env(env);
    }
    print_test_result(success);
    return success ? 0 : 1;
}
