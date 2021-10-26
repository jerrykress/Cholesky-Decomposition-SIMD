.DEFAULT_GOAL := compile

CC = ../compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++

PROG_SEQ  = main_seq
PROG_SIMD = main_neon

RUN_PROG = ${PROG_SIMD}
TEST_FILE = input_8x8.txt

compile:
	@echo "Compiling to aarch64..."
	${CC} ${PROG_SEQ}.cpp -o ${PROG_SEQ}.elf
	${CC} ${PROG_SIMD}.cpp -o ${PROG_SIMD}.elf

run:
	@echo "Running compiled binary..."
	chmod +x ${RUN_PROG}.elf && ./${RUN_PROG}.elf ${TEST_FILE}

clean:
	@echo "Cleaning built binaries and generated input files..."
	rm *.elf
	rm *.txt