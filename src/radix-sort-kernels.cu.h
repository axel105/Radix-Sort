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

    // Assuming histogram is filled with 0
    extern __shared__ uint32_t histogram[];

    // TODO: 
    // - Sort the block locally
    // - Use shared memory
    for (uint32_t i = 0; i < elem_pthread; ++i) { // handle elements per thread numbner of elements 
        uint32_t next = globalId * elem_pthread + i; // index of element to be handled
        if (next < in_size) { // check that we're in bounds
            // get b bits corresponding to the current iteration
            uint32_t rank = 
                (keys[next] >> (it * bits)) & (hist_size - 1); 
            atomicAdd(&g_hist[rank], 1); // atomically increase the value of the bucket at index rank
        }
    }

    // copy from shared memory to global memory
    //__syncthreads();
    //if (globalId < hist_size) {
    //    printf("histogram[%d]: %d\n", globalId, histogram[globalId]);
    //    g_hist[globalId] = histoamgram[globalId]; // copy histogram value to global memory
    //}
}

/**
 * This kernel transposes a given Matrix in a coalesced fascion
 * source: https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/transpose/doc/MatrixTranspose.pdf
*/
__global__ void transposeCoalesced(float *odata,
            float *idata, int width, int height)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
  int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;
  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
    tile[threadIdx.y+i][threadIdx.x] =
      idata[index_in+i*width];
  }
  __syncthreads();
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
    odata[index_out+i*height] =
      tile[threadIdx.x][threadIdx.y+i];
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

/*
 * This kernel transposes the matrix
*/
__global__ void transpose(int* in, int width, int height, int* out){
    int numPerThread = 4;

    __shared__ double tile[16][16];
    int i_n = blockIdx.x * tileSize + threadIdx.x;
    int i_m = blockIdx.y * tileSize + threadIdx.y; // <- threadIdx.y only between 0 and 7

    // Load matrix into tile
    // Every Thread loads in this case 4 elements into tile.
    int i;
    for (i = 0; i < tileSize; i += BLOCK_ROWS){
        if(i_n < n  && (i_m+i) < m){
            tile[threadIdx.y+i][threadIdx.x] = matIn[(i_m+i)*n + i_n];
        }
    }
    __syncthreads();

    i_n = blockIdx.y * TILE_DIM + threadIdx.x; 
    i_m = blockIdx.x * TILE_DIM + threadIdx.y;

    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS){
        if(i_n < m  && (i_m+i) < n){
            matTran[(i_m+i)*m + i_n] = tile[threadIdx.x][threadIdx.y + i]; // <- multiply by m, non-squared!

        }
    }
}


#endif // !RADIX_KERNEL
