rm -rf radix_bench;
futhark cuda radix_bench.fut &&\
futhark dataset --i32-bounds=0:9999999 -g [10000]i32 | ./radix_bench -t /dev/stderr -r 1 > /dev/null
futhark dataset --i32-bounds=0:9999999 -g [100000]i32 | ./radix_bench -t /dev/stderr -r 1 > /dev/null
futhark dataset --i32-bounds=0:9999999 -g [1000000]i32 | ./radix_bench -t /dev/stderr -r 1 > /dev/null
futhark dataset --i32-bounds=0:9999999 -g [10000000]i32 | ./radix_bench -t /dev/stderr -r 1 > /dev/null
futhark dataset --i32-bounds=0:9999999 -g [100000000]i32 | ./radix_bench -t /dev/stderr -r 1 > /dev/null
