#include <stdio.h>
#include "utils.cu.h"
#include "radix-sort-kernels.cu.h"

int main(int argc, char **argv){
    if (argc != 2) {
        printf("Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
    const int H = 15;

    //Allocate and Initialize Host data with random values
    // CPU
    int* h_keys  = (int*) malloc(N*sizeof(int));
    int* h_keys_res  = (int*) malloc(16*sizeof(int));
    randomInitNat(h_keys, N, H);

    //Allocate and Initialize Device data
    //GPU
    int* d_keys_in;
    int* d_keys_out;
    //cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    //cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));

    cudaMalloc((void**) &d_keys_in,  N * sizeof(int));
    cudaMemcpy(d_keys_in, h_keys, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_keys_out, N * sizeof(int));

    int b = 4;
    int elementPerThread = 4;
    //const int HistSize = 1 << b;

    int block_size  = 256;
    int num_blocks = 
        (N + block_size * elementPerThread - 1) / (block_size * elementPerThread);
    sortAndWriteHistogram<<<num_blocks, block_size>>>(d_keys_in, d_keys_out, b, elementPerThread, N, 2);

    //cudaCheckError();

    cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //cudaCheckError();

    printf("input array: [");
    for(int i = 0; i < N; i++){
        printf("%d, ",  h_keys[i]);
    }
    printf("]\n");

    printf("res_array: [");
    for(int i = 0; i < N; i++){
        printf("%d, ",  h_keys_res[i]);
    }
    printf("]\n");

    int success = 0;
    return success ? 0 : 1;
}
