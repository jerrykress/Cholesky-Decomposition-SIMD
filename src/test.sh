#!/bin/bash

rm test_results.txt

for i in {16..32}
do
   echo "Testing size $i"
   make generate DIM=$i
   make run DIM=$i
   make check
done