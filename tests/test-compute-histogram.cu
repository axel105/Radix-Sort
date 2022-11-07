#include "cpu_functions.cu.h"
#include "test_utils.cu.h"

// from src
#include "helper.cu.h"
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "radix-sort.cu.h"
#include "types.cu.h"
#include "utils.cu.h"

#define NUM_BLOCKS_NEEDED 17
#define BLOCK_SIZE 256
#define ELEMENT_PER_THREAD 4
#define INPUT_MAX_SIZE BLOCK_SIZE * ELEMENT_PER_THREAD * NUM_BLOCKS_NEEDED

bool valid_histogram(kernel_env env) {
    for (uint32_t it = 0; it < size_in_bits<uint32_t>(); it += env->bits) {
        uint32_t iteration = it / env->bits;
        // compute histogram on CPU
        compute_histogram(env->h_keys, env->h_hist, env->bits, env->h_keys_size,
                          iteration);

        // compute histogram on GPU
        compute_histogram_local(env, iteration);
        uint32_t hist_size = d_hist_size(env), histogram[hist_size], scanned_hist[hist_size];
        reduce_d_hist(env, histogram);

        if (!equal(env->h_hist, histogram, env->number_classes)) {
            if (TEST_DEBUG) {
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

bool sorted_properties(kernel_env env) {
    //printf("- Testing that the keys are sorted at each steps...\n");
    // compute histogram on GPU
    uint32_t iteration = 0;
    compute_histogram_local(env, iteration);
    uint32_t keys[env->d_keys_size];
    uint32_t keys_size_for_one_block = env->block_size * env->elem_pthread;
    d_keys(env, keys);
    // checking that the keys array is sort on a block level
    for(int index = 0; index < env->d_keys_size - 1; ++index){
        if(index > 0 && index % keys_size_for_one_block) continue; 
        uint32_t current_key = keys[index] >> (iteration * env->bits) & (env->number_classes-1);
        uint32_t next_key = keys[index + 1] >> (iteration * env->bits) & (env->number_classes-1);
        if(current_key > next_key) { 
            fprintf(stderr, "keys[%d]: %d, keys[%d]: %d\n",keys[index], keys[index-1]);
            return false;
        }
    }
    return true;
}

bool test(){
    bool success = true;
    printf("*** Testing compute histogram kernel properties ***\n");
    printf("- histogram computed is valid\n");
    printf("- The keys are sorted per block\n");
    for (uint32_t number_keys = 2; number_keys < INPUT_MAX_SIZE;
         number_keys*=2) {

        const uint32_t block_size = 256, elem_pthread = 4, bits = 4,
                       max_value = number_keys/2;

        kernel_env env = new_kernel_env(block_size, elem_pthread, bits,
                                    number_keys, max_value);
        printf("\rTesting with %d keys", number_keys);

        success &= test(valid_histogram, env);
        success &= test(sorted_properties, env);

        free_env(env);
    }
    printf("\n");
    return success;

}

bool test_with_keys(const uint32_t number_keys){
    bool success = true;
    printf("*** Testing compute histogram kernel properties ***\n");
    printf("- histogram computed is valid\n");
    printf("- The keys are sorted per block\n");

    const uint32_t block_size = 256, elem_pthread = 4, bits = 4,
                    max_value = number_keys/2;

    kernel_env env = new_kernel_env(block_size, elem_pthread, bits,
                                number_keys, max_value);
    printf("\rTesting with key: %d", number_keys);

    success &= test(valid_histogram, env);
    success &= test(sorted_properties, env);

    free_env(env);
    printf("\n");
    return success;

}

int main(int argc, char **argv) {
    
    // bool success = test_with_keys(16384);
    bool success = test();
    print_test_result(success);
    return success ? 0 : 1;
}
