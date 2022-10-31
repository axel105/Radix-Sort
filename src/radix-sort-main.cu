#include <stdio.h>
#include "utils.cu.h"
#include "radix-sort-kernels.cu.h"
#include "radix-test.cu.h"
#include "radix-sort-cpu.cu.h"

// Parse the input arguments
int parse_args(int argc, char **argv){
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    return atoi(argv[1]);
}


int main(int argc, char **argv){
    const int in_size = parse_args(argc, argv);
    //test_compute_histogram_cpu(in_size, 15, 4);

    const uint32_t bits = 4, max_value = 15, 
    elem_pthread = 4, num_thread = 256;

    //bool success = test_kernel1(in_size, bits, max_value, elem_pthread, num_thread);
    //success ? fprintf(stdout, "VALID!\n"): fprintf(stdout, "INVALID!\n");

    bool success2 = test_kernel2(in_size, bits, max_value, elem_pthread, num_thread);

    return success2 ? 0 : 1;
}
