#ifndef RADIX_KERNEL
#define RADIX_KERNEL
#include "cub/cub.cuh"
#include "types.cu.h"

/**
 * This kernel loads and sorts it's b-bit of it's tile on on-chip memory and
 * writes the histogram and sorted tile to global memory
 *
 * @keys: Array to sort
 * @g_hist: Output array (representing the histogram)
 * @bits: number of bits too look at
 * @elem_pthread: number of keys process by one thread at the time
 * @in_size: Size of @keys
 * @hist_size: Size of @g_hist
 * @it: iteration number
 */
__global__ void compute_histogram(uint32_t *keys, uint32_t *g_hist,
                                  uint32_t bits, uint32_t elem_pthread,
                                  uint32_t in_size, uint32_t hist_size,
                                  uint32_t it) {
    // TODO:
    // - Sort the block locally
    // - See if we can gain performance by avoiding using atomicAdd
    // - Eventually avoid rewrite if conditions...?
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threadId = threadIdx.x;

    // Assuming histogram is filled with 0
    extern __shared__ uint32_t histogram[];
    if (globalId < hist_size) {
        histogram[globalId] = 0;
    }
    // TODO: change thread acess to keys to be more coalesced
    // t0 -> keys[0]
    // t1 -> keys[1]
    // t2 -> keys[2]
    // ...
    // t31 -> keys[31]
    //
    // thread_access: (blockIdx.x * blockDim.x * elem_pthread) + (blockDim.x *
    // i) + threadIdx.x
    // (0 * 256 * 4) + (256 * 0 ) + 0
    // (0 * 256 * 4) + (256 * 1 ) + 0
    // (0 * 256 * 4) + (256 * 2 ) + 0 512
    // (0 * 256 * 4) + (256 * 3 ) + 0
    //
    // (0 * 256 * 4) + (256 * 0 ) + 1
    // (0 * 256 * 4) + (256 * 1 ) + 1
    // (0 * 256 * 4) + (256 * 2 ) + 1
    // (0 * 256 * 4) + (256 * 3 ) + 1
    for (uint32_t i = 0; i < elem_pthread;
         ++i) {  // handle elements per thread numbner of elements
        uint32_t next =
            globalId * elem_pthread + i;  // index of element to be handled
        if (next < in_size) {             // check that we're in bounds
            // get b bits corresponding to the current iteration
            uint32_t rank = (keys[next] >> (it * bits)) & (hist_size - 1);
            atomicAdd(&histogram[rank], 1);  // atomically increase the value of
                                             // the bucket at index rank
        }
    }

    // copy from shared memory to global memory
    __syncthreads();
    if (threadId < hist_size) {
        int offset =
            blockIdx.x * hist_size + threadId;  // calculate position in global
                                                // memory for histogram value
        g_hist[offset] =
            histogram[threadId];  // copy histogram value to global memory
    }
}

__global__ void exampleKernel(int *out) {
    // printf("Executing Example Kernel\n");
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<int, 256> BlockScan;
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    int thread_data[4];
    for (int i = 0; i < 4; i++) {
        thread_data[i] = threadIdx.x;
        printf("---threadId %d\n", threadIdx.x);
    }
    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
    for (int i = 0; i < 4; i++) {
        printf("***thread_data[%d] = %d\n", i, thread_data[i]);
        out[i + threadIdx.x] = thread_data[i];
    }
}

// TODO:
// - Sort the block locally
// - See if we can gain performance by avoiding using atomicAdd
// - Eventually avoid rewrite if conditions...?
__global__ void compute_histogram_sort(uint32_t *d_keys, uint32_t *g_hist,
                                       uint32_t *g_scan_hist, uint32_t bits,
                                       uint32_t elem_pthread,
                                       uint32_t d_keys_size,
                                       uint32_t num_classes, uint32_t it) {
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t width = blockDim.x * elem_pthread;
    uint32_t size = blockDim.x * elem_pthread * num_classes;

    // TODO:  change size
    // shared_histogram[number_classes][block_size*elem_pthread]
    extern __shared__ uint16_t flag_arrays[];
    __shared__ uint32_t shared_keys[256 * 4];
    __shared__ uint32_t shared_sorted_keys[256 * 4];
    __shared__ uint32_t shared_histogram[16];
    __shared__ uint32_t shared_scan_hist[16];

    // Initialize flag_arrays with 0..?
    //
    for (int i = threadIdx.x; i < size; i += blockDim.x) flag_arrays[i] = 0;
    __syncthreads();

    // make a flag array from keys
    // TODO: change thread acess to keys to be more coalesced
    // t0 -> keys[0]
    // t1 -> keys[1]
    // t2 -> keys[2]
    // ...
    // t31 -> keys[31]
    //
    // thread_access: (blockIdx.x * blockDim.x * elem_pthread) + (blockDim.x *
    // i) + threadIdx.x
    // -- First Block
    // (0 * 256 * 4) + (256 * 0 ) + 0 =  0
    // (0 * 256 * 4) + (256 * 1 ) + 0 =  256
    // (0 * 256 * 4) + (256 * 2 ) + 0 =  512
    // (0 * 256 * 4) + (256 * 3 ) + 0 =  768
    //
    // (0 * 256 * 4) + (256 * 0 ) + 1 =
    // (0 * 256 * 4) + (256 * 1 ) + 1 =  257
    // (0 * 256 * 4) + (256 * 2 ) + 1 =  513
    // (0 * 256 * 4) + (256 * 3 ) + 1 =  769
    //
    // -- Second Block
    // (1 * 256 * 4) + (256 * 0 ) + 0 =  1024
    // (1 * 256 * 4) + (256 * 1 ) + 0 =  1280
    // (1 * 256 * 4) + (256 * 2 ) + 0 =  1536
    // (1 * 256 * 4) + (256 * 3 ) + 0 =  1792
    //
    // (1 * 256 * 4) + (256 * 0 ) + 1 =  1025
    // (1 * 256 * 4) + (256 * 1 ) + 1 =  1281
    // (1 * 256 * 4) + (256 * 2 ) + 1 =  1537
    // (1 * 256 * 4) + (256 * 3 ) + 1 =  1793
    //
    // TODO: store keys array in shared memory, for future read

    uint32_t nth_bits = it * bits;
    uint32_t class_range = num_classes - 1;
    for (uint32_t i = 0; i < elem_pthread;
         ++i) {  // handle elements per thread numbner of elements
        uint32_t block_offset = blockIdx.x * blockDim.x * elem_pthread;
        uint32_t block_index = blockDim.x * i + threadIdx.x;
        uint32_t index = block_offset + block_index;
        if (index < d_keys_size) {  // check that we're in bounds
            // get b bits corresponding to the current iteration
            uint32_t key = d_keys[index];
            uint32_t rank = (key >> nth_bits) & class_range;
            flag_arrays[rank * width + block_index] =
                1;                           // hist[rank][next] = 1
            shared_keys[block_index] = key;  // save keys in shared mem
        }
    }

    // if(globalId == 0){
    //     printf("Shared keys\n[");
    //     for(int i = 0; i < width; i++){
    //         if(i > 0 && i % 32 == 0) printf("\n");
    //         printf("%d, ", shared_keys[i]);
    //     }
    //     printf("]\n\n");
    // }

    // debug
    // if(globalId == 0){
    //     printf("FLAG ARRAY\n");
    //     for(int i = 0; i < size; i++){
    //         if(i > 0 && i % width == 0) printf("\n");
    //         printf("%d, ", flag_arrays[i]);
    //     }
    //     printf("\n\n");
    // }

    __syncthreads();  // waiting to flag the entire array

    // scan the flag array
    // Specialize BlockScan for a 1D block of 128 threads of type int
    typedef cub::BlockScan<uint16_t, 256> BlockScan16;
    __shared__ typename BlockScan16::TempStorage temp_storage16;
    // TODO: make thread process 4 elements, thread_data => thread_data[4]
    for (int rank = 0; rank < num_classes; ++rank) {
        uint16_t thread_data[4];
        // load the data from shared
        for (int i = 0; i < elem_pthread; ++i) {
            int index = (rank * width) + threadIdx.x * elem_pthread + i;
            // printf("***threadIdx: %d, index: %d\n", threadIdx.x, index);
            thread_data[i] = flag_arrays[index];
        }
        __syncthreads();  // maybe optional

        // scan
        BlockScan16(temp_storage16).InclusiveSum(thread_data, thread_data);

        __syncthreads();  // required (by documentation)

        // write back in shared_memory
        for (int i = 0; i < elem_pthread; ++i) {
            int index = (rank * width) + threadIdx.x * elem_pthread + i;
            flag_arrays[index] = thread_data[i];
        }
    }

    // debug
    __syncthreads();  // waiting to flag the entire array

    // if(globalId == 0){
    //     printf("SCANNED FLAG ARRAY\n");
    //     for(int i = 0; i < size; i++){
    //         if(i > 0 && i % width == 0) printf("\n");
    //         printf("%d, ", flag_arrays[i]);
    //     }
    //     printf("\n\n");
    // }

    // sort the array localy
    // [4, 3, 4, 3, 4]
    // c_x [......] -> (0) number_element of rank x
    // c_x [......] -> (1)
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // c_x [......]
    // sort[......]
    //
    // 1. exclusive scan of histogram (last row of scan_hist)
    // 2. (scan_flag[rank][index]-1) + scan_hist[rank]
    typedef cub::BlockScan<uint32_t, 16> BlockScan32;
    __shared__ typename BlockScan32::TempStorage temp_storage32;
    if (threadIdx.x < num_classes) {
        int last_row = (threadIdx.x * width) + (width - 1);
        uint32_t thread_data = flag_arrays[last_row];
        shared_histogram[threadIdx.x] = thread_data;  // saving histogram
        BlockScan32(temp_storage32).ExclusiveSum(thread_data, thread_data);
        shared_scan_hist[threadIdx.x] = thread_data;
    }

    __syncthreads();  // waiting for the scan hist

    // if(globalId == 0){
    //    printf("hist\n[");
    //    for(int i = 0; i < num_classes; i++){
    //        printf("%d, ", shared_histogram[i]);
    //    }
    //    printf("]\n\n");

    //    printf("scan hist\n[");
    //    for(int i = 0; i < num_classes; i++){
    //        printf("%d, ", shared_scan_hist[i]);
    //    }
    //    printf("]\n\n");
    //}

    // sort array keys locally
    for (uint32_t i = 0; i < elem_pthread;
         ++i) {  // handle elements per thread numbner of elements
        uint32_t block_offset = blockIdx.x * blockDim.x * elem_pthread;  // 0
        uint32_t block_index = blockDim.x * i + threadIdx.x;             //
        uint32_t index = block_offset + block_index;                     // 11
        if (index < d_keys_size) {  // check that we're in bounds
            uint32_t key =
                shared_keys[block_index];  // save keys in shared mem //11
            uint32_t rank = (key >> nth_bits) & class_range;
            uint32_t number_elem_previous_rank = shared_scan_hist[rank];
            uint32_t nth_element = flag_arrays[rank * width + block_index];
            uint32_t sorted_position =
                nth_element - 1 + number_elem_previous_rank;
            // printf("***threadIdx.x: %d, index: %d, key: %d, rank:%d,
            // num_elem_prev: %d, nth_elem: %d, sorted_position:%d\n",
            // threadIdx.x, index, key, rank, number_elem_previous_rank,
            // nth_element, sorted_position);
            shared_sorted_keys[sorted_position] = key;
        }
    }
    /// for(int i = 0; i < elem_pthread; ++i){
    ///    uint32_t index_key = threadIdx.x + i * blockDim.x;
    ///    if(index_key + blockDim.x * elem_pthread * blockIdx.x < size){
    ///        uint32_t key = shared_keys[index_key]; // save keys in shared mem
    ///        uint32_t rank = (key >> nth_bits) & class_range;
    ///        uint32_t number_elem_previous_rank = shared_scan_hist[rank];
    ///        uint32_t nth_element = flag_arrays[rank * width + index_key];
    ///        uint32_t sorted_position = nth_element - 1 +
    ///        number_elem_previous_rank; shared_sorted_keys[sorted_position] =
    ///        key;
    ///    }
    ///}

    __syncthreads();  // waiting  for the sort

    // if(globalId == 0){
    //     printf("SORTED ARRAY\n[");
    //     for(int i = 0; i < width; i++){
    //         if(i > 0 && i % 32 == 0) printf("\n");
    //         printf("%d, ", shared_sorted_keys[i]);
    //     }
    //     printf("\n\n\n");
    // }

    // TODO:  write to global memroy
    // - scan histogram
    // - sorted array
    // - histogram
    for (uint32_t i = 0; i < elem_pthread;
         ++i) {  // handle elements per thread numbner of elements
        uint32_t block_offset = blockIdx.x * blockDim.x * elem_pthread;
        uint32_t block_index = blockDim.x * i + threadIdx.x;
        uint32_t index = block_offset + block_index;
        if (index < d_keys_size) {  // check that we're in bounds
            d_keys[index] =
                shared_sorted_keys[block_index];  // save keys in shared mem
        }
    }
    // copy histogram to global memory
    if (threadIdx.x < num_classes) {
        int offset =
            blockIdx.x * num_classes +
            threadIdx
                .x;  // calculate position in global memory for histogram value
        g_hist[offset] = shared_histogram[threadIdx.x];  // copy histogram value
                                                         // to global memory
        g_scan_hist[offset] = shared_scan_hist[threadIdx.x];
    }
}

/*
 * This kernel transposes a matrix
 *
 * @odata: transpose matrix
 * @idata: matrix to transpose
 */
#define TILE_DIM 16
#define BLOCK_ROWS 4
__global__ void transposeNaive(uint32_t *odata, uint32_t *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

/*
 * This kernel transposes a matrix
 *
 * @odata: transpose matrix
 * @idata: matrix to transpose
 */
__global__ void transpose(uint32_t *odata, uint32_t *idata) {
    // no boundary checking needed as we spawn with a block size of 256, and our
    // histogram array size is a multiple of 256

    __shared__ uint32_t tile[256];

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t index = 16 * (threadIdx.x % 16) + (threadIdx.x / 16);

    tile[index] = idata[globalId];
    __syncthreads();

    odata[globalId] = tile[threadIdx.x];
}

/**
 * This kernel uses a scan histogram to recompute the position of
 * each element of keys
 *
 * @keys: input array to sort
 * @output: output array
 * @hist: scanned histogram
 * @bits: number of bits considered for one keys
 * @elem_pthread: number of elements managed by one thread
 * @key_size: size of @keys
 * @hist_size: size of @hist
 * @it: number of iteration,
 *  (corresponding to the bits being processed; eg. it = 0 => first 4 bits)
 *
 */
__global__ void scatter(uint32_t *keys, uint32_t *output, uint32_t *hist,
                        uint32_t bits, uint32_t elem_pthread, uint32_t key_size,
                        uint32_t hist_size, uint32_t it) {
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (!globalId) {
        printf("keys GPU: [");
        for (uint32_t i = 0; i < key_size; ++i) {
            printf("%d, ", keys[i]);
        }
        printf("]\n");

        printf("hist GPU: [");
        for (uint32_t i = 0; i < hist_size; ++i) {
            printf("%d, ", hist[i]);
        }
        printf("]\n");
    }

    for (uint32_t i = 0; i < elem_pthread;
         ++i) {  // handle elements per thread numbner of elements
        uint32_t next =
            globalId * elem_pthread + i;  // index of element to be handled
        if (next < key_size) {            // check that we're in bounds
            // get b bits corresponding to the current iteration
            uint32_t rank = (keys[next] >> (it * bits)) & (hist_size - 1);
            int index =
                atomicAdd(&hist[rank], -1);  // atomically decrease the value of
                                             // the bucket at index rank
            // printf("next: %d, key: %d, rank: %d, index: %d\n", next,
            // keys[next], rank, index);
            output[index] = keys[next];
        }
    }
}

__global__ void array_from_scan(uint32_t *odata, uint32_t *idata, int hist_size,
                                int b) {
    if (threadIdx.x < (1 << b)) {
        int width = hist_size / (1 << b);

        int index = width * (threadIdx.x + 1) - 1;
        odata[threadIdx.x] = idata[index];
    }
}

__global__ void scatter_coalesced(uint32_t *keys, uint32_t *output,
                                  uint32_t *hist_T_scanned, uint32_t *hist,
                                  uint32_t bits, uint32_t elem_pthread,
                                  uint32_t key_size, uint32_t hist_size,
                                  uint32_t it) {
    // scan the histogram and save the data in shared memory

    uint32_t num_buckets = 1 << bits;
    __shared__ uint32_t local_hist[16];
    __shared__ uint32_t scanned_hist[16];

    if (threadIdx.x < num_buckets) {  // copy histogram corresponding to block
                                      // into shared memory
        int tmp = hist[num_buckets * blockIdx.x + threadIdx.x];

        // Specialize BlockScan for a 1D block of 16 threads of type int
        typedef cub::BlockScan<int, 16> BlockScan;

        // Allocate shared memory for BlockScan
        __shared__ typename BlockScan::TempStorage temp_storage;

        // Obtain a segment of consecutive items that are blocked across threads

        // Collectively compute the block-wide exclusive prefix sum
        BlockScan(temp_storage).ExclusiveSum(tmp, tmp);
        scanned_hist[threadIdx.x] = tmp;
    }

    __syncthreads();

    // assumptions for this to work:
    // - presorted array on the block level
    // - scan of the histogram must be exclusive
    // - scan transpose histogram must be exclusive

    for (int i = 0; i < 4; ++i) {  // each thread scatters 4 elements
        // TODO: read in coalesced fascion (see gilles kernel 1)
        uint32_t next_local = i * blockDim.x + threadIdx.x;
        uint32_t next_global =
            blockIdx.x * blockDim.x * elem_pthread + next_local;

        if (next_global < key_size) {
            uint32_t element = keys[next_global];
            uint32_t rank = (element >> (it * bits)) & (hist_size - 1);
            uint32_t number_of_elems_in_previous_ranks_locally =
                scanned_hist[rank];
            int local_rank_offset =
                next_local -
                number_of_elems_in_previous_ranks_locally;  // the nth element
                                                            // with this rank in
                                                            // our block

            // arr: [0, 1]  , hit: [1, 1], scan_hist: [0, 1]
            // index: 1, num_elem_previous_local = 1, local_rank_offset = 0

            // debug
            if (local_rank_offset < 0) {
                printf("--------- local rank offset smaller than 0 ---------");
            }

            // hist_T_scanned[rank][blockIdx.x]
            int global_rank_index_start =
                hist_T_scanned[rank * (hist_size / num_buckets) + blockIdx.x];
            int global_index = global_rank_index_start + local_rank_offset;
            output[global_index] = element;
        }
    }
}

/**
 * This kernel transposes a given Matrix in a coalesced fascion
 * source:
 * https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/transpose/doc/MatrixTranspose.pdf
 */
//__global__ void transposeCoalesced(float *odata,
//            float *idata, int width, int height)
//{
//  __shared__ float tile[TILE_DIM][TILE_DIM];
//  int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
//  int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
//  int index_in = xIndex + (yIndex)*width;
//  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//  int index_out = xIndex + (yIndex)*height;
//  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//    tile[threadIdx.y+i][threadIdx.x] =
//      idata[index_in+i*width];
//  }
//  __syncthreads();
//  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
//    odata[index_out+i*height] =
//      tile[threadIdx.x][threadIdx.y+i];
//  }
//}

#endif  // !RADIX_KERNEL
