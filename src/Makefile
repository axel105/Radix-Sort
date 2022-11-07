OPT_FLAGS = -O3
CUB = ../cub-code/cub-1.8.0
EXEC = main
CFLAGS = -g

all: $(EXEC)

$(EXEC): main.cu types.cu.h kernel_env.cu.h radix-sort.cu.h kernels.cu.h helper.cu.h
	nvcc $(CFLAGS) -I$(CUB)/cub -o $@ $< 

.PHONY: clean

clean:
	rm -f $(EXEC)
