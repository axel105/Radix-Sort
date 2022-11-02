#ifndef KERNEL_ENV
#define KERNEL_ENV
#include "types.cu.h"

void log_d_keys(kernel_env env){
    uint32_t res[env->d_keys_size];
    cudaMemcpy(res, env->d_keys, env->d_keys_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    log_vector(res, env->d_keys_size);
}

void log_d_hist(kernel_env env){
    uint32_t res[env->d_hist_size];
    cudaMemcpy(res, env->d_hist, env->d_hist_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    log_vector_with_break(res, env->d_hist_size, env->number_classes);
}

void debug_env_gpu_settings(kernel_env env){
    fprintf(stderr, "--- GPU Settings ---\n");
    fprintf(stderr, "block_size = %d\n", env->block_size);
    fprintf(stderr, "num_blocks = %d\n", env->num_blocks);
    fprintf(stderr, "element per thread = %d\n", env->elem_pthread);
    fprintf(stderr, "bits = %d\n", env->bits);
    fprintf(stderr, "--------------------\n");
}

void debug_env_data_props(kernel_env env){
    fprintf(stderr, "--- Data Props   ---\n");
    fprintf(stderr, "number classes = %d\n", env->number_classes);
    fprintf(stderr, "number of keys = %d\n", env->h_keys_size);
    fprintf(stderr, "maximum value = %d\n", env->max_value);
    fprintf(stderr, "d_keys_size = %d\n", env->d_keys_size);
    fprintf(stderr, "d_hist_size = %d\n", env->d_hist_size);
    fprintf(stderr, "------------------\n");
}

void debug_env_data(kernel_env env){
    fprintf(stderr, "---    Data    ---\n");
    fprintf(stderr, "host input array:\n");
    log_vector(env->h_keys, env->h_keys_size);
    fprintf(stderr, "device input array:\n");
    log_d_keys(env);
    fprintf(stderr, "device hist:\n");
    log_d_hist(env);
    fprintf(stderr, "------------------\n");
}

void debug_env(kernel_env env){
    fprintf(stderr, "*** Kernel environment ***\n");
    fprintf(stderr, "{\n");
    debug_env_gpu_settings(env);
    debug_env_data_props(env);
    debug_env_data(env);
    fprintf(stderr, "}\n");
}


/**
 * Create (allocate) and initialize a kernel_env
 *
 * @block_size: number of thread in one block
 * @elem_pthread: number of element processed by one thread
 * @bits: number of bits considered for one key
 * @number_keys: number of keys to generate (input array)
 * @max_value: maximum value a key can have
 */
kernel_env new_kernel_env(uint32_t block_size, uint32_t elem_pthread,
                            uint32_t bits, uint32_t number_keys,
                            uint32_t max_value){

    kernel_env env = (kernel_env) malloc(sizeof(struct Kernel_env));

    // *** GPU Settings
    env->block_size = block_size;
    env->num_blocks = 
        (number_keys + block_size * elem_pthread - 1) / (block_size * elem_pthread); 
    env->elem_pthread = elem_pthread;
    env->bits = bits;

    // *** Data attributes
    env->number_classes = 1 << env->bits;
    env->max_value = max_value;
    // - host data (CPU)
    env->h_keys_size = number_keys;
    env->h_hist_size = env->number_classes;
    // - device data (GPU)
    env->d_keys_size = number_keys;
    env->d_hist_size = env->num_blocks * env->number_classes + 
        (env->number_classes - 
            (env->num_blocks % env->number_classes)) * env->number_classes;
 

    // *** Allocate memory and initialize the data
    // - Allocate and init on host (CPU)
    env->h_keys = (uint32_t*) malloc(env->h_keys_size * sizeof(uint32_t));
    env->h_hist = (uint32_t*) malloc(env->h_hist_size * sizeof(uint32_t));
    randomInitNat(env->h_keys, env->h_keys_size, env->max_value);
    // - Allocate and init on device (GPU)
    cudaMalloc((void**) &env->d_keys,  env->d_keys_size * sizeof(uint32_t));
    cudaMemcpy(env->d_keys, env->h_keys, env->h_keys_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &env->d_hist,  env->d_hist_size * sizeof(uint32_t));

    if(DEBUG) debug_env(env);

    return env;
}

void free_env(kernel_env env){
    if(DEBUG) fprintf(stderr, "Cleaning kernel env: %p!\n", env);
    if(env == NULL) return;
    if(env->h_keys != NULL) free(env->h_keys);
    if(env->h_hist != NULL) free(env->h_hist);
    cudaFree(env->d_keys);
    cudaFree(env->d_hist);
}

#endif //!KERNEL_ENV
