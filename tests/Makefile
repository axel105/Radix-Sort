CUB = ../cub-code/cub-1.8.0
RADIX = ../src

CFLAGS = -I$(CUB)/cub -I$(RADIX) 
OPT_FLAGS = -O3

TEST = test-kernels

all: compile-all test-all

compile-all: test-kernels.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) test-compute-histogram.cu &&\
	nvcc $(CFLAGS) test-kernels.cu


test-all:
	./test-compute-histogram

$(TEST): test-kernels.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) -o $@ $<

test-compute-histogram: test-compute-histogram.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) $(OPTFLAGS) -o $@ $< && ./test-compute-histogram

.PHONY: clean

clean:
	rm -f $(TEST) test-compute-histogram
