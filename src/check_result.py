# Script to check the squared sum difference of two output files.
# Usage python3 check_result.py <filename1> <filename2>

import numpy as np
import sys

# get input
args = sys.argv
fn1 = args[1]
fn2 = args[2]
dim = args[3]

# buffer
arr1 = []
arr2 = []

# read file 1
with open(fn1, "r") as f:
    for line in f.readlines():
        for token in line.split():
            num = float(token)
            if num != float("NAN"):
                arr1.append(num)
            else:
                print("Check failed! NaN found in result 1.")
                sys.exit()

# read file 2
with open(fn2, "r") as f:
    for line in f.readlines():
        for token in line.split():
            num = float(token)
            if num != float("NAN"):
                arr2.append(num)
            else:
                print("Check failed! NaN found in result 2.")
                sys.exit()

# Check size
if len(arr1) != len(arr2):
    print("Check failed! Length of the output do not match.")
    sys.exit()

# Calculate diff
diff = np.array(arr1) - np.array(arr2)
total = np.sum(diff ** 2)
print("**** Check passed ****!\n Squared sum difference: ", total)

# Append to testing results
with open("test_results.txt", 'a') as f:
    f.write(str(dim) + ' ' + str(total) + '\n')