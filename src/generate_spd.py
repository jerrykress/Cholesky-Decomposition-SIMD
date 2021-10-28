# A simple script that generate symmetrical positive definite matrix given the size
# usage: python3 generate_spd.py <size>

import numpy as np
import sys
from termcolor import colored

# get size
args = sys.argv
if(len(args) < 2):
    print (colored("Error! Generation dimension not specified. Abort", 'red'))
    sys.exit()
dim = int(args[1])

# check valid dim
if type(dim) != type(1) or dim < 3:
    print (colored("Input dimension type error. Abort", 'red'))
    sys.exit()

# create matrix
A = np.random.randint(50, size=(dim, dim))
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

msg = "Successfully generated new file: " + filename
print(colored(msg, "green"))