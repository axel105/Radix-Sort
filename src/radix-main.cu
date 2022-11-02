#include <stdio.h>
#include "test-kernels.cu.h"
#include "types.cu.h"
#include "kernel_env.cu.h"

// Parse the input arguments
int parse_args(int argc, char **argv){
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    return atoi(argv[1]);
}


int main(int argc, char **argv){
    const int number_keys = parse_args(argc, argv);

    const uint32_t block_size = 256, elem_pthread = 4, 
    bits = 4, max_value = 16;

    kernel_env env = new_kernel_env(block_size, elem_pthread,
                                    bits, number_keys, max_value);

    if (test_compute_histogram(env))
        printf("VALID !\n");
    else printf("INVALID !\n");

    test_transpose(env);
    test_scan(env);


    free_env(env);


    return 0;
    
}
