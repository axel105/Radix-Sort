#ifndef RADIX_KERNEL
#define RADIX_KERNEL

typedef unsigned int uint32_t; 


/**
 * Step 1 + 2: This kernel loads and sorts it's b-bit of it's tile on on-chip memory and 
 * writes the histogram and sorted tile to global memory
*/
__global__ void sortAndWriteHistogram(int* d_inp, int* d_out, int b, int elementsPerThread, int arrLen, int iteration) {
    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int histSize = 16;

    __shared__ int histogram[16];

    histogram[threadId%(histSize)] = 0;
    __syncthreads();

    // TODO: Sort the block locally


    for (int i=0; i < elementsPerThread; ++i) { // handle elements per thread numbner of elements 
        int next = globalId*4 + i; // index of element to be handled
        if (next < arrLen) { // check that we're in bounds
            int rank = (d_inp[next] >> (iteration*b)) & (histSize-1); // get b bits corresponding to the current iteration
            atomicAdd(&histogram[rank], 1); // atomically increase the value of the bucket at index rank
        }
    }
    __syncthreads();

    // TODO: try to keep the histogram in local memory, for use in next kernel
    // 

    if (threadId < histSize) {
        int offset = blockIdx.x * histSize + threadId; // calculate position in global memory for histogram value
        d_out[offset] = histogram[threadId]; // copy histogram value to global memory
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
