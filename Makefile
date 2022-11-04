all: main

main:
	cd src && make clean && make && ./main 10

test:
	cd tests && make clean && make && ./test-kernels 1024

clean:
	cd src && make clean && cd .. && cd tests && make clean
