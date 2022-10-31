typedef struct KERNEL_ENV{
    uint32_t *h_keys;
    uint32_t *h_hist;

    uint32_t* d_keys;
    uint32_t* d_hist;

    uint32_t num_blocks;
    uint32_t num_thread;
}kernel_env;

kernel_env* init_kernel_env(uint32_t in_size, uint32_t max_value, 
                            uint32_t number_classes, uint32_t num_thread){

    kernel_env *env = malloc(sizeof(kernel_env));

    //Allocate memory on host and initialize the input array
    uint32_t* h_keys = (uint32_t*) malloc(in_size * sizeof(uint32_t));
    uint32_t* h_hist = (uint32_t*) malloc(number_classes * sizeof(uint32_t));
    randomInitNat(h_keys, in_size, max_value);

    // GPU Cuda properties, number of blocks, and thread
    uint32_t num_blocks = 
        (in_size + num_thread * elem_pthread - 1) / (num_thread * elem_pthread); 

    //Allocate memory on GPU and initialize it
    uint32_t* d_keys;
    uint32_t* d_hist;
    cudaMalloc((void**) &d_keys,  in_size * sizeof(uint32_t));
    cudaMalloc((void**) &d_hist, number_classes * num_blocks * sizeof(uint32_t));
    cudaMemcpy(d_keys, h_keys, in_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    env->h_keys = h_keys;
    env->h_hist = h_hist;
    env->d_keys = d_keys;
    env->d_hist = d_hist;
    env->num_blocks = num_blocks;
    env->num_thread = num_thread;

}
