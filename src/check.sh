#!/bin/bash

for i in {128..256}
do
   echo "Checking size $i"
   python3 check_result.py out/main_seq_out_$i.txt out/main_neon_out_$i.txt $i
done