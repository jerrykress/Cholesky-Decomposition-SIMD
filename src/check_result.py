# Script to check the squared sum difference of two output files.
# Usage python3 check_result.py <filename1> <filename2>

import numpy as np
import sys
from termcolor import colored

######################################################################
#                 Output
######################################################################

# get input
args = sys.argv
dim = args[1]
fn1 = args[2]
fn2 = args[3]

# buffer
arr1 = []
arr2 = []

# read matrix A
with open(fn1, "r") as f:
    for line in f.readlines():
        for token in line.split():
            num = float(token)
            if num != float("NAN"):
                arr1.append(num)
            else:
                print("Check failed! NaN found in result 1.")
                sys.exit()

# read matrix B
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
total = np.nansum(diff ** 2)
percent = total / np.nansum(arr1) * 100
msg = "[DIFF] " + str(total) + "(" + str(percent) + "%)"
print(colored(msg, "green"))


######################################################################
#                 Perf
######################################################################

# get input
fn3 = args[4]
fn4 = args[5]

# buffer
time_a = []
time_b = []
duration_a = 0
duration_b = 0

# get time a
with open(fn3, "r") as f:
    for line in f.readlines():
        for token in line.split():
            time_a.append(token)
duration_a = float(time_a[0])

# get time b
with open(fn4, "r") as f:
    for line in f.readlines():
        for token in line.split():
            time_b.append(token)
duration_b = float(time_b[0])

######################################################################
#                 Write Out
######################################################################

# Append to testing results
with open("test_results.txt", "a") as f:
    f.write(
        str(dim)
        + " "
        + str(total)
        + " "
        + str(percent)
        + " "
        + str(duration_a)
        + " "
        + str(duration_b)
        + "\n"
    )
