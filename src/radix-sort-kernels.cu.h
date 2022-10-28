#ifndef RADIX_KERNEL
#define RADIX_KERNEL

typedef unsigned int uint32_t; 

/**
 * Naive memcpy kernel, for the purpose of comparing with
 * a more "realistic" bandwidth number.
 
__global__ void naiveMemcpy(float* d_out, float* d_inp, const uint32_t N) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        d_out[gid] = d_inp[gid];
    }
}
*/

/**
 * Step 1: This kernel loads and sorts its tile in on chip memory
*/
__global__ void loadSort(int* d_inp) {

}

/**
 * Step 2: This kernel writes the histogram and sorted tile to global memory
*/
__global__ void writeHistogramAndData(int* d_out) {

}

/**
 * Step 3: This kernel performs a prefix sum on all the histogram tables to compute global digit offsets
*/
__global__ void prefixSum(int* d_inp) {
    
}

/**
 * Step 1: This kernel scatters the elements in their intended position
*/
__global__ void scatter(int* d_inp) {
    
}


#endif // !RADIX_KERNEL
