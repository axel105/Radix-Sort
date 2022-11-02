#include <stdio.h>
#include "utils.cu.h"
#include "radix-sort-kernels.cu.h"
#include "radix-test.cu.h"
#include "radix-sort-cpu.cu.h"
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

    free_env(env);


    return 0;
    
}
