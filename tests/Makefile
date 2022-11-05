CUB = ../cub-code/cub-1.8.0
RADIX = ../src

CFLAGS = -I$(CUB)/cub -I$(RADIX) -g
OPT_FLAGS = -O3

TEST = test-kernels

all: $(TEST)

$(TEST): test-kernels.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) -o $@ $<

test-compute-histogram: test-compute-histogram.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) -o $@ $< && ./test-compute-histogram

.PHONY: clean

clean:
	rm -f $(TEST)
