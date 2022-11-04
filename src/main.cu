#include <stdio.h>
#include "kernel_env.cu.h"
#include "kernels.cu.h"
#include "radix-sort.cu.h"
#include "types.cu.h"
#include "utils.cu.h"

int main(int argc, char **argv) {
    const int number_keys = parse_args(argc, argv);

    const uint32_t block_size = 256, elem_pthread = 4, bits = 4, max_value = 16;

    kernel_env env =
        new_kernel_env(block_size, elem_pthread, bits, number_keys, max_value);

    radix_sort(env);
    log_result(env);

    free_env(env);

    return 0;
}
