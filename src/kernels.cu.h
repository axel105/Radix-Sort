#ifndef RADIX_KERNEL
#define RADIX_KERNEL
#include "cub/cub.cuh"
#include "types.cu.h"

/**
 * This kernel loads and sorts it's b-bit of it's tile on on-chip memory and
 * writes the histogram and sorted tile to global memory
 *
 * @d_keys: Array to sort and array to write back array pre-sorted on the block level
 * @g_hist: Output array (representing the histogram)
 * @g_scan_hist: Output array (representing the scanned histogram)
 * @bits: number of bits too look at
 * @elem_pthread: number of keys process by one thread at the time
 * @d_keys_size: Size of @keys
 * @num_classes: Size of @g_hist
 * @it: iteration number
 */
__global__ void compute_histogram_sort(uint32_t *d_keys, uint32_t *g_hist,
                                       uint32_t *g_scan_hist, uint32_t bits,
                                       uint32_t elem_pthread,
                                       uint32_t d_keys_size,
                                       uint32_t num_classes, uint32_t it) {
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t width = blockDim.x * elem_pthread;
    uint32_t size = blockDim.x * elem_pthread * num_classes;

    // shared_histogram[number_classes][block_size*elem_pthread]
    extern __shared__ uint16_t flag_arrays[];
    __shared__ uint32_t shared_keys[256 * 4];
    __shared__ uint32_t shared_sorted_keys[256 * 4];
    __shared__ uint32_t shared_histogram[16];
    __shared__ uint32_t shared_scan_hist[16];

    // Initialize flag_arrays with 0
    for (int i = threadIdx.x; i < size; i += blockDim.x) flag_arrays[i] = 0;
    __syncthreads();

    // make a flag array from keys

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

    __syncthreads();  // waiting to flag the entire array

    // scan the flag array
    // Specialize BlockScan for a 1D block of 256 threads of type int
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

    // exclusive scan of histogram (last row of scan_hist)
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
            shared_sorted_keys[sorted_position] = key;
        }
    }
    __syncthreads();  // waiting  for the sort

    // write back sorted block in coalesced fascion
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

__global__ void scatter_coalesced(uint32_t *keys, uint32_t *output,
                                  uint32_t *hist_T_scanned, uint32_t *hist,
                                  uint32_t *hist_rowwise_scanned, uint32_t bits,
                                  uint32_t elem_pthread, uint32_t key_size,
                                  uint32_t hist_size, uint32_t it) {
    // scan the histogram and save the data in shared memory

    uint32_t num_buckets = 1 << bits;
    __shared__ uint32_t local_hist[16];
    __shared__ uint32_t scanned_hist[16];

    if (threadIdx.x < num_buckets) {  // copy histogram corresponding to block
                                      // into shared memory
        local_hist[threadIdx.x] = hist[num_buckets * blockIdx.x + threadIdx.x];
        scanned_hist[threadIdx.x] =
            hist_rowwise_scanned[num_buckets * blockIdx.x + threadIdx.x];
    }

    __syncthreads();

    // assumptions for this to work:
    // - presorted array on the block level
    // - scan of the histogram must be exclusive
    // - scan transpose histogram must be exclusive

    for (int i = 0; i < 4; ++i) {  // each thread scatters 4 elements
        uint32_t next_local = i * blockDim.x + threadIdx.x;
        uint32_t next_global =
            blockIdx.x * blockDim.x * elem_pthread + next_local;

        if (next_global < key_size) {
            uint32_t element = keys[next_global];
            uint32_t rank = (element >> (it * bits)) & (num_buckets - 1);
            uint32_t number_of_elems_in_previous_ranks_locally =
                scanned_hist[rank];
            int local_rank_offset =
                next_local -
                number_of_elems_in_previous_ranks_locally;  // the nth element
                                                            // with this rank in
                                                            // our block

            // hist_T_scanned[rank][blockIdx.x]
            int global_rank_index_start =
                hist_T_scanned[rank * (hist_size / num_buckets) + blockIdx.x];
            int global_index = global_rank_index_start + local_rank_offset;
            output[global_index] = element;
        }
    }

    // debug to remove later
    __syncthreads();
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0){
        printf("\n??????????????????? Printing on GPU ????????????????\n");
        printf("\nNON SCANNED HISTOGRAM\n[");
        for(int i = 0; i < hist_size; ++i) {
            if(i > 0 && i % num_buckets == 0) printf("\n");
            printf(" %d, ", hist[i]);
        }
        printf("\nSCANNED TRANSPOSED HISTOGRAM\n[");
        for(int i = 0; i < hist_size; ++i) {
            if(i > 0 && i % (hist_size/16) == 0) printf("\n");
            printf(" %d, ", hist_T_scanned[i]);
        }
        printf("]\n????????????????????????????????????????????????????\n");
    }
}
#endif  // !RADIX_KERNEL