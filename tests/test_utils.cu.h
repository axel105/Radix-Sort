#ifndef TEST_UTILS
#define TEST_UTILS
#include <stdbool.h>
#include "types.cu.h"

bool equal(uint32_t* vec1, uint32_t* vec2, const uint32_t size){
    for(uint32_t i = 0; i < size; ++i){
        if(vec1[i] != vec2[i]) return false;
    }
    return true;
}

#endif // !TEST_UTILS
