#ifndef TYPE
#define TYPE
#define DEBUG 1

typedef unsigned int uint32_t;
typedef unsigned short uint16_t;

typedef struct Kernel_env {
    // general data attributes
    uint32_t number_classes;
    uint32_t h_keys_size;
    uint32_t h_hist_size;
    uint32_t max_value;
    uint32_t *h_keys;
    uint32_t *h_hist;

    // GPU data attributes
    uint32_t *d_keys;
    uint32_t *d_hist;
    uint32_t *d_hist_scan;
    uint32_t *d_hist_transpose;
    uint32_t *d_scan_res;
    uint32_t *d_output;
    uint32_t d_keys_size;
    uint32_t d_hist_size;

    // GPU settings
    uint32_t num_blocks;
    uint32_t block_size;
    uint32_t elem_pthread;
    uint32_t bits;
} * kernel_env;

// -------------- kernel_env functions -------------------
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
                          uint32_t max_value);

void free_env(kernel_env env);

kernel_env copy(kernel_env env);

//***** Getters
uint32_t number_classes(kernel_env env);

uint32_t d_hist_size(kernel_env env);

uint32_t d_keys_size(kernel_env env);

uint32_t *d_keys(kernel_env env);

uint32_t *d_hist(kernel_env env);

uint32_t bits(kernel_env env);

uint32_t elem_pthread(kernel_env env);

void d_vec(uint32_t *source, uint32_t *dest, uint32_t size);

void d_keys(kernel_env env, uint32_t *output);

void d_hist(kernel_env env, uint32_t *output);

void d_hist_transpose(kernel_env env, uint32_t *output);

void d_scan_result(kernel_env env, uint32_t *output);

void d_output(kernel_env env, uint32_t *output);

void reduce_d_hist(kernel_env env, uint32_t *output);
//--------------- DEBUG FUNCTIONS ---------------------
void log_d_keys(kernel_env env);

void log_d_hist(kernel_env env);

void log_d_hist_transpose(kernel_env env);

void log_reduce_d_hist(kernel_env env);

void log_scan_result(kernel_env env);

void log_output_result(kernel_env env);

void debug_env_gpu_settings(kernel_env env);

void debug_env_data_props(kernel_env env);

void debug_env_data(kernel_env env);

void debug_env(kernel_env env);

#endif  // TYPE
