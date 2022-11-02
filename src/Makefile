OPT_FLAGS = -O3
CUB=../cub-code/cub-1.8.0

all: radix-sort main

radix-sort: radix-sort-main.cu radix-sort-kernels.cu.h radix-test.cu.h radix-sort-cpu.cu.h
	nvcc -I$(CUB)/cub -o radix-sort-main radix-sort-main.cu 

main: radix-main.cu radix-sort-kernels.cu.h radix-test.cu.h radix-sort-cpu.cu.h types.cu.h kernel_env.cu.h
	nvcc -I$(CUB)/cub -o main radix-main.cu 


.PHONY: clean

clean:
	rm -f radix-sort-main main
