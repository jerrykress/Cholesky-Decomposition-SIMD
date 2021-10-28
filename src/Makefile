.DEFAULT_GOAL := compile

CC = ../compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++

PROG_SEQ  = main_seq
PROG_SIMD = main_neon

TEST_FILE = input_${DIM}x${DIM}.txt

generate:
	@echo "Generating a random input file of size ${DIM}"
	python3 generate_spd.py ${DIM}

compile:
	@echo "Compiling to aarch64..."
	${CC} ${PROG_SEQ}.cpp -o ${PROG_SEQ}.elf
	${CC} ${PROG_SIMD}.cpp -o ${PROG_SIMD}.elf

run:
	@echo "Running compiled binary..."
	chmod +x ${PROG_SEQ}.elf && ./${PROG_SEQ}.elf ${TEST_FILE}
	chmod +x ${PROG_SIMD}.elf && ./${PROG_SIMD}.elf ${TEST_FILE}

check:
	@echo "Compare output..."
	python3 check_result.py ${PROG_SEQ}_out.txt ${PROG_SIMD}_out.txt

clean:
	@echo "Cleaning built binaries and generated input files..."
	rm *.elf
	rm *.txt
