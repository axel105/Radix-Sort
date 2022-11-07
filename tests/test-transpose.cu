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

bool valid_transpose(kernel_env env) {
    transpose_histogram(env);
    uint32_t t_hist[env->d_hist_size];
    uint32_t hist[env->d_hist_size];
    d_hist_transpose(env, t_hist);
    d_hist(env, hist);
    
    uint32_t width = env->elem_pthread * env->block_size;

    for(int i = 0; i < env->d_hist_size; ++i){
        //if(hist[i] != t_hist[i * width 
        //find formula
        
    }
    return true;
}


int main(int argc, char **argv) {
    bool success = true;

    printf("*** Testing histogram tranpostion ***\n");

    for (uint32_t number_keys = 2; number_keys < INPUT_MAX_SIZE;
         number_keys*=2) {

        const uint32_t block_size = 256, elem_pthread = 4, bits = 4,
                       max_value = number_keys/2;

        kernel_env env = new_kernel_env(block_size, elem_pthread, bits,
                                    number_keys, max_value);
        printf("\rTesting with %d keys", number_keys);

        success &= test(valid_histogram, env);

        free_env(env);
    }
    printf("\n");
    print_test_result(success);
    return success ? 0 : 1;
}
