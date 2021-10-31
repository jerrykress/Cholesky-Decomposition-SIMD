#!/bin/bash

rm test_results.txt

for i in {128..256}
do
   echo "Testing size $i"
#    make generate DIM=$i
   make run DIM=$i
#    make check
done