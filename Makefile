all: main

main:
	cd src && make clean && make && ./main 10

test:
	cd tests && make clean && make && ./test-kernels 1024

format: 
	make format-test && make format-src

format-test:
	clang-format -i tests/*.cu.h tests/*.cu

format-src:
	clang-format -i src/*.cu.h src/*.cu


clean:
	cd src && make clean && cd .. && cd tests && make clean
