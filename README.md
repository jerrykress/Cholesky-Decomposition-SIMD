# MSc-Project-2021

This project implements Cholesky Decomposition in C++ on Arm Neon SIMD platforms. The purpose is to compare the differences in performance and power consumption in order to demonstrate the benefits and drawbacks of using SIMD on matrices of various sizes.

## Compiler

> gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu

Get in terminal

> `wget https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz`

_Or download newer versions from_ [Arm developer website](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads)

Extract using

> ` tar -Jxf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz`

## How to use

Compile all versions of binaries

> `make`

Generate a random input file

> `make generate DIM=<SIZE>`

Run program for a matrix size

> `make run DIM=<SIZE>`

Compare the output from NEON against squential

> `make check <FILE_1> <FILE_2>`

Run an automatic test on a range of sizes and generate a report file

> `make test DIM1=<RANGE_BEGIN> DIM2=<RANGE_END>`
