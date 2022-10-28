OPT_FLAGS = -O3

all: radix-sort

radix-sort: radix-sort-main.cu radix-sort-kernels.cu.h
	nvcc -o radix-sort-main radix-sort-main.cu
	#./radix-sort-main.cu

clean:
	rm -f radix-sort-main
