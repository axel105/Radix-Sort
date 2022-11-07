CUB = ../cub-code/cub-1.8.0
RADIX = ../src

CFLAGS = -I$(CUB)/cub -I$(RADIX) 
OPT_FLAGS = -O3

TEST = test-kernels

all: compile-all test-all

compile-all: test-kernels.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) -o test-radix-sort test-radix-sort.cu &&\
	nvcc $(CFLAGS) -o test-compute-histogram test-compute-histogram.cu &&\
	nvcc $(CFLAGS) -o test-kernels test-kernels.cu

test-all:
	./test-compute-histogram &&\
	./test-radix-sort

$(TEST): test-kernels.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) -o $@ $<

test-compute-histogram: test-compute-histogram.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) $(OPTFLAGS) -o $@ $< && ./test-compute-histogram

test-radix-sort: test-radix-sort.cu test_utils.cu.h cpu_functions.cu.h
	nvcc $(CFLAGS) $(OPTFLAGS) -o $@ $< && ./test-radix-sort


.PHONY: clean

clean:
	rm -f $(TEST) test-compute-histogram
