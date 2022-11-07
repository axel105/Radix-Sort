#ifndef TEST_UTILS
#define TEST_UTILS
#include <stdbool.h>
#include <stdio.h>
#include "types.cu.h"

#define TEST_DEBUG 1
#undef DEBUG
#define DEBUG 0

bool equal(uint32_t* vec1, uint32_t* vec2, const uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        if (vec1[i] != vec2[i]) return false;
    }
    return true;
}

bool test(bool (*test_function)(kernel_env), kernel_env env) {
    bool success = test_function(env);
    //success ? fprintf(stderr, "Test Passed!\n")
    //        : fprintf(stderr, "Test FAILED!\n");
    return success;
}

void print_test_result(bool success) {
    success ? fprintf(stderr, "All test passed!\n")
            : fprintf(stderr, "Tests failed!\n");
}

#endif  // !TEST_UTILS
