OPT_FLAGS = -O3

all: radix-sort

radix-sort: radix-sort-main.cu radix-sort-kernels.cu.h radix-test.cu.h radix-sort-cpu.cu.h
	nvcc -o radix-sort-main radix-sort-main.cu 

.PHONY: clean

clean:
	rm -f radix-sort-main
