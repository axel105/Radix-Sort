#ifndef KERNEL_ENV
#define KERNEL_ENV
#include "types.cu.h"
#include "utils.cu.h"
#include "helper.cu.h"

//TODO: FIX COPY function


kernel_env new_kernel_env(uint32_t block_size, uint32_t elem_pthread,
                            uint32_t bits, uint32_t number_keys,
                            uint32_t max_value){

    kernel_env env = (kernel_env) malloc(sizeof(struct Kernel_env));

    // *** GPU Settings
    env->block_size = block_size;
    env->elem_pthread = elem_pthread;
    env->bits = bits;
    env->num_blocks = 
        (number_keys + block_size * elem_pthread - 1) 
        / (block_size * elem_pthread); 

    // *** Data attributes
    env->number_classes = 1 << bits;
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
    env->h_keys = (uint32_t*) calloc(env->h_keys_size, sizeof(uint32_t));
    env->h_hist = (uint32_t*) calloc(env->h_hist_size, sizeof(uint32_t));
    randomInitNat(env->h_keys, env->h_keys_size, env->max_value);

    // - Allocate and init on device (GPU)
    cudaSucceeded(cudaMalloc((void**) &env->d_keys,  env->d_keys_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(env->d_keys, env->h_keys, env->h_keys_size * sizeof(uint32_t), 
        cudaMemcpyHostToDevice));

    cudaSucceeded(cudaMalloc((void**) &env->d_hist,  env->d_hist_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &env->d_hist_scan,  env->d_hist_size * sizeof(uint32_t)));

    cudaSucceeded(cudaMalloc((void**) &env->d_output,  env->d_keys_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &env->d_hist_transpose,  env->d_hist_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &env->d_scan_res,  env->number_classes * sizeof(uint32_t)));

    cudaDeviceSynchronize();
    cudaCheckError();

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

kernel_env copy(kernel_env env){
    if(env == NULL) return NULL;
    kernel_env env_copy = (kernel_env) malloc(sizeof(struct Kernel_env));

    // *** GPU Settings
    env_copy->block_size = env->block_size;
    env_copy->num_blocks = env->num_blocks;
    env_copy->elem_pthread = env->elem_pthread;
    env_copy->bits = env->bits;

    // *** Data attributes
    env_copy->number_classes = env->number_classes;
    env_copy->max_value = env->max_value;
    // - host data (CPU)
    env_copy->h_keys_size = env->h_keys_size;
    env_copy->h_hist_size = env->h_hist_size;
    // - device data (GPU)
    env_copy->d_keys_size = env->d_keys_size;
    env_copy->d_hist_size = env->d_hist_size; 

    // *** Allocate memory and initialize the data
    // - Allocate and init on host (CPU)
    env_copy->h_keys = (uint32_t*) malloc(env->h_keys_size * sizeof(uint32_t));
    env_copy->h_hist = (uint32_t*) malloc(env->h_hist_size * sizeof(uint32_t));
    memcpy(env_copy->h_keys, env->h_keys, env->h_keys_size * sizeof(uint32_t));
    memcpy(env_copy->h_hist, env->h_hist, env->h_hist_size * sizeof(uint32_t));

    // - Allocate and init on device (GPU)
    cudaSucceeded(cudaMalloc((void**) &env_copy->d_keys,  env->d_keys_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(env_copy->d_keys, env->d_keys, env->d_keys_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    cudaSucceeded(cudaMalloc((void**) &env_copy->d_hist,  env->d_hist_size * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(env_copy->d_hist, env->d_hist, env->d_hist_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();

    cudaCheckError();

    return env_copy;

}

uint32_t number_classes(kernel_env env){
    return env->number_classes;
}

uint32_t d_hist_size(kernel_env env){
    return env->d_hist_size;
}

uint32_t d_keys_size(kernel_env env){
    return env->d_keys_size;
}

uint32_t* d_keys(kernel_env env){
    return env->d_keys;
}

uint32_t* d_hist(kernel_env env){
    return env->d_hist;
}

uint32_t bits(kernel_env env){
    return env->bits;
}

uint32_t elem_pthread(kernel_env env){
    return env->elem_pthread;
}

void d_vec(uint32_t *source, uint32_t *dest, uint32_t size){
    cudaMemcpy(dest, source, size * sizeof(uint32_t), 
                cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void d_keys(kernel_env env, uint32_t *output){
    d_vec(env->d_keys, output, env->d_keys_size);
}

void d_hist(kernel_env env, uint32_t *output){
    d_vec(env->d_hist, output, env->d_hist_size);
}

void d_hist_transpose(kernel_env env, uint32_t *output){
    d_vec(env->d_hist_transpose, output, env->d_hist_size);
}

void d_scan_result(kernel_env env, uint32_t *output){
    d_vec(env->d_scan_res, output, env->number_classes);
}

void d_output(kernel_env env, uint32_t *output){
    d_vec(env->d_output, output, env->d_keys_size);
}

void reduce_d_hist(kernel_env env, uint32_t *output){
    d_hist(env, output);
    for(uint32_t i = env->number_classes; i < env->d_hist_size; ++i){
        uint32_t index = i & (env->number_classes - 1);
        output[index] += output[i];
    }
}
//--------------- DEBUG FUNCTIONS ---------------------
void log_d_keys(kernel_env env){
    uint32_t size = env->d_keys_size, res[size];
    d_keys(env, res);
    log_vector(res, size);
}

void log_d_hist(kernel_env env){
    uint32_t res[env->d_hist_size];
    cudaMemcpy(res, env->d_hist, env->d_hist_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    log_vector_with_break(res, env->d_hist_size, env->number_classes);
}

void log_d_hist_scan(kernel_env env){
    uint32_t res[env->d_hist_size];
    cudaMemcpy(res, env->d_hist_scan, env->d_hist_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    log_vector_with_break(res, env->d_hist_size, env->number_classes);
}

void log_d_hist_transpose(kernel_env env){
    uint32_t res[env->d_hist_size];
    cudaMemcpy(res, env->d_hist_transpose, env->d_hist_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    log_vector_with_break(res, env->d_hist_size, env->number_classes);
}

void log_reduce_d_hist(kernel_env env){
    uint32_t res[env->d_hist_size];
    reduce_d_hist(env, res);
    log_vector(res, env->number_classes);
}

void log_scan_result(kernel_env env){
    uint32_t res[env->number_classes];
    d_scan_result(env, res);
    log_vector(res, env->number_classes);
}

void log_output_result(kernel_env env){
    uint32_t res[env->d_keys_size];
    d_output(env, res);
    log_vector(res, env->d_keys_size);
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

void log_result(kernel_env env){
    fprintf(stdout, "input array:\n");
    log_d_keys(env);
    fprintf(stdout, "output array:\n");
    log_output_result(env);


}


#endif //!KERNEL_ENV
