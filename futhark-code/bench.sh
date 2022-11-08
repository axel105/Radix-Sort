echo "--------------------------------"
echo "Testing with C"
rm -rf radix_bench;
futhark c radix_bench.fut &&\
echo "Testing with i32-bounds=0:9999 -g[700000]"
futhark dataset --i32-bounds=0:9999 -g[700000] | ./radix_bench -t /dev/stderr -r 10 > /dev/null

echo "--------------------------------"
echo "Testing with CUDA"
rm -rf radix_bench;
futhark cuda radix_bench.fut &&\
echo "Testing with i32-bounds=0:9999 -g[700000]"
futhark dataset --i32-bounds=0:9999 -g[700000] | ./radix_bench -t /dev/stderr -r 10 > /dev/null
