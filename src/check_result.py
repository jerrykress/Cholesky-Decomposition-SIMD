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
            if not np.isnan(num):
                arr1.append(num)
            else:
                msg = "Check failed! NaN found in file 1: " + fn1
                print(colored(msg, "red"))
                sys.exit()

# read matrix B
with open(fn2, "r") as f:
    for line in f.readlines():
        for token in line.split():
            num = float(token)
            if not np.isnan(num):
                arr2.append(num)
            else:
                msg = "Check failed! NaN found in file 2: " + fn2
                print(colored(msg, "red"))
                sys.exit()

# Check size
if len(arr1) != len(arr2):
    print(colored("Check failed! Length of the output do not match.", "red"))
    sys.exit()

# Calculate diff
diff = np.array(arr1) - np.array(arr2)
total = np.nansum(diff ** 2)
percent = total / np.nansum(arr1) * 100
msg = "[DIFF] " + str(total) + " (" + str(percent) + "%)"
if percent > 0.0005:
    print(colored(msg, "yellow"))
else:
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
