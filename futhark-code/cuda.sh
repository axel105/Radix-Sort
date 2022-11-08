echo "--------------------------------"
echo "Testing with CUDA"
rm -rf radix_bench;
futhark cuda radix_bench.fut &&\
echo "Testing with i32-bounds=0:10000000 -g[10000000]"
futhark dataset --i32-bounds=0:100000000000 -g [1000000]i32 | ./radix_bench -t /dev/stderr -r 1 > /dev/null
