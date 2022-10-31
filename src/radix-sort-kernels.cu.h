#ifndef RADIX_KERNEL
#define RADIX_KERNEL

typedef unsigned int uint32_t; 


/**
 * Step 1 + 2: This kernel loads and sorts it's b-bit of it's tile on on-chip memory and 
 * writes the histogram and sorted tile to global memory
 *
 * @keys Array to sort
 * @g_hist Output array (representing the histogram)
 * @bits number of bits too look at
 * @elem_pthread number of keys process by one thread at the time
 * @in_size Size of @keys
 * @hist_size Size of @g_hist
 * @it iteration number
*/
__global__ void compute_histogram(uint32_t* keys, uint32_t* g_hist, 
                                      uint32_t bits, uint32_t elem_pthread, 
                                      uint32_t in_size,uint32_t hist_size, 
                                      uint32_t it) {
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threadId = threadIdx.x;


    // Assuming histogram is filled with 0
    extern __shared__ uint32_t histogram[];
    //printf("threadIdx.x : %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);

    // TODO: 
    // - Sort the block locally
    // - Use shared memory
    for (uint32_t i = 0; i < elem_pthread; ++i) { // handle elements per thread numbner of elements 
        uint32_t next = globalId * elem_pthread + i; // index of element to be handled
        if (next < in_size) { // check that we're in bounds
            // get b bits corresponding to the current iteration
            uint32_t rank = 
                (keys[next] >> (it * bits)) & (hist_size - 1); 
            atomicAdd(&histogram[rank], 1); // atomically increase the value of the bucket at index rank
            //if (blockIdx.x == 1)
            //printf("--- histogram[%d]: %d\n", rank, histogram[rank]);
        }
    }

    // copy from shared memory to global memory
    __syncthreads();
    //if (globalId < hist_size) {
    //    g_hist[globalId] = histogram[globalId]; // copy histogram value to global memory
    //}
    if (threadId < hist_size) {
       int offset = blockIdx.x * hist_size + threadId; // calculate position in global memory for histogram value
       //printf("*** histogram[%d]: %d\n", threadId, histogram[threadId]);
       printf("*** offset: %d, threadId: %d\n", offset, threadId);
       g_hist[offset] = histogram[threadId]; // copy histogram value to global memory
    }
                            
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
