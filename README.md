# Cholesky Decomposition on SIMD

This project implements Cholesky Decomposition in C++ using Arm Neon, Intel AVX-256 intrinsics and OpenMP.
The purpose is to evaluate the advantage of running the algorithm on SIMD platforms and compare the differences in performance between architectures.
(This repo is part of a MSc thesis at University of Glasgow)

## Pre-Install Configuration

Clone the repo. Navigate to the **src** directory. Before installing the program, specify the target platform in the Makefile option below. This will change how the sequential reference code is compiled. Do not pass this in as command line parameter.

```bash
# running on AArch64 architecture
TARGET_ARCH = aarch64

# running on x86 architecture
TARGET_ARCH = x86_64
```

## Installation

After the target has been specified, run the following command. This will download and install the default compiler then build all the binaries.

```bash
make install
```

---

## Pre-Testing Configuration

Before conducting any test, modify the benchmark program to be tested. **Reference program should always be set to sequential**. Do not pass these in as command line parameters.

```bash
# available programs
PROG_SEQ  = main_seq
PROG_NEON = main_neon
PROG_AVX  = main_avx
PROG_NEON_OMP = main_neon_omp
PROG_AVX_OMP  = main_avx_omp
# reference program
PROG_1 = ${PROG_SEQ}
# benchmark program (change this)
PROG_2 = ${PROG_AVX_OMP}
```

## Automatic Testing

Once the desired configuration is completed, run the following command to start automatic ranged testing. This will automatically generate the required input files, run both sequential and the chosen benchmark programs at each size in the specified range, collect the results and produce a graph.

Some testing options are available in the section below.

```bash
make test DIM1=<start> DIM2=<end> [Options]
```

For example to run a test from 64 to 1024 with an interval of 16 using double precision floats, type the following command.

```bash
make test DIM1=64 DIM2=1024 STEP=16 PRECISION=2
```

## Manual Testing

```bash
# extract pre-generated input files
make extract

# generate a single input file
make generate DIM=<size>

# generate a batch of input files
make generate_batch DIM1=<start> DIM2=<end> STEP=<interval>

# run a single test on a specific size
make run DIM=<size>

# invoke evaluation of test result
make check DIM=<size>

# generate graph from test_results.txt
make graph

# delete all input, output files and built binaries
make clean
```

---

## Options

```bash
# specify the size of intervals in a ranged automatic testing
# the value should be equal or greater than 1
# default: 1
STEP=1

# specify using single or double floating point precision.
# PRECISION=1: use 32-bit precision in all programs
# PRECISION=2: use 64-bit precision in all programs
# default: 1
PRECISION=1

# generate new random input files in automatic testing
# GENERATE_NEW=0: use previous generated files
# GENERATE_NEW=1: generate new every time
# default: 1
GENERATE_NEW=1

# keep output and result files from a previous test. This is useful when performing tests in segments.
# KEEP_HISTORY=0: delete previous output before testing
# KEEP_HISTORY=1: keep previous output before testing
# default: 0
KEEP_HISTORY=0

# GCC compiler flag
# This is not used by default. However you could pass in gcc options such as vectorisation flag through this option.
CC_VEC_FLAG=[gcc-options]
```

## Uninstall

Uninstall compiler and perform a clean, leaving only the original source code.

```bash
make uninstall
```

## Recommended Content
[Neon Intrinsics References](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=6093,14,3171,3159,3144,4341,3814,2174,2174,4914,4242,4317,6138,6127&text=mm256_)

[Arm GNU Toolchain for the A-profile Architecture](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads)

[Arm GNU Compiler Options](https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html)

## License

[MIT](https://choosealicense.com/licenses/mit/)
