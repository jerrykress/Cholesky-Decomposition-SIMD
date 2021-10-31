#!/bin/bash

for i in {128..256}
do
   echo "Generating size $i"
   make generate DIM=$i
done