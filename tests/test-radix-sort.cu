#include "cpu_functions.cu.h"
#include "test_utils.cu.h"

// from src
#include "helper.cu.h"
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "radix-sort.cu.h"
#include "types.cu.h"
#include "utils.cu.h"

#define NUM_BLOCKS_NEEDED 3
#define INPUT_MAX_SIZE 1024 * 16 * NUM_BLOCKS_NEEDED

bool sorted_array(kernel_env env) {
    radix_sort(env);
    uint32_t res[env->d_keys_size];
    d_output(env, res);
    for (int i = 0; i < env->d_keys_size - 1; ++i) {
        if (res[i] > res[i + 1]) {
            debug_env_data_props(env);
            fprintf(stderr, "res[%i]: %d, res[%i]: %d\n", i, res[i], i+1, res[i + 1]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    bool success = true;
    printf("\n*** Testing radix sort ***\n");
    for (uint32_t number_keys = 2; number_keys < INPUT_MAX_SIZE;
         number_keys*=2) {

        const uint32_t block_size = 256, elem_pthread = 4, bits = 4,
                       max_value = number_keys/2;

        kernel_env env = new_kernel_env(block_size, elem_pthread, bits,
                                    number_keys, max_value);
        printf("\rTesting with %d keys", number_keys);

        success &= test(sorted_array, env);

        free_env(env);
    }
    printf("\n");
    print_test_result(success);
    return success ? 0 : 1;
}
