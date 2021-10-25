# A simple script that generate symmetrical positive definite matrix given the size
# usage: python3 generate_spd.py <size>

import numpy as np
import sys

# get size
args = sys.argv
dim = int(args[1])

# check valid dim
if type(dim) != type(1) or dim < 3:
    print("Input dimension type error. Abort")
    sys.exit()

# create matrix
A = np.random.rand(dim, dim)
B = np.dot(A, A.transpose())

# get filename
filename = "input_" + str(dim) + "x" + str(dim) + ".txt"
print("Creating SPD Matrix:", filename, "\n")

# write to file
with open(filename, "w") as f:
    for row in B:
        line = ""
        for num in row:
            line += str(num) + " "
        f.write(line[:-1] + "\n")

# verify file
with open(filename, "r") as f:
    for line in f.readlines():
        print(line)
