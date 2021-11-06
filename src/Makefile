.DEFAULT_GOAL := compile
.SILENT:test

CYAN=\033[0;36m
NC=\033[0m

CC = ../compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++

DCC = ../compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-objdump

PROG_SEQ  = main_seq
PROG_SIMD = main_neon

IN_PATH = input
OUT_PATH = output

RESULT_FILE = test_results.txt
TEST_FILE = input_${DIM}x${DIM}.txt
STEP ?= 1

generate:
	@echo "Generating random input size ${DIM}"
	@rm -rf ${IN_PATH}
	@mkdir -p ${IN_PATH}
	@python3 generate_spd.py ${DIM}
	@mv ${TEST_FILE} ${IN_PATH}

compile:
	@echo "Compiling to aarch64..."
	${CC} ${PROG_SEQ}.cpp -o ${PROG_SEQ}.elf
	${CC} -march=armv8-a -O3 ${PROG_SIMD}.cpp -o ${PROG_SIMD}.elf

compile_fast:
	${CC} -O3 ${PROG_SEQ}.cpp -o ${PROG_SEQ}_O3.elf
	${CC} -O3 ${PROG_SIMD}.cpp -o ${PROG_SIMD}_O3.elf

run:
	@mkdir -p ${OUT_PATH}
	@echo "SEQ @ size=${DIM}"
	@chmod +x ${PROG_SEQ}.elf && ./${PROG_SEQ}.elf ${IN_PATH}/${TEST_FILE}
	@echo "SIMD @ size=${DIM}"
	@chmod +x ${PROG_SIMD}.elf && ./${PROG_SIMD}.elf ${IN_PATH}/${TEST_FILE}

test:
	@echo "Testing sizes from ${DIM1} up to ${DIM2}"
	@touch *
	@rm -rf ${IN_PATH}
	@rm -rf ${OUT_PATH}
	@rm -f ${RESULT_FILE}
	@i=${DIM1}; while [ "$$i" -le ${DIM2} ]; do \
		clear; echo "${CYAN}Testing $$i / ${DIM2} ${NC}"; make -s generate DIM=$$i; make -s run DIM=$$i; make -s check DIM=$$i; i=$$((i + ${STEP}));\
  	done
	@make -s graph


check:
	@echo "Compare output..."
	@python3 check_result.py ${OUT_PATH}/main_seq_out_${DIM}.txt ${OUT_PATH}/main_neon_out_${DIM}.txt ${DIM}

graph:
	@python3 graph.py

inspect:
	@${DCC} -lSd {PROG_SIMD}.elf > O1.txt 

clean:
	@echo "Cleaning built binaries and generated input files..."
	@rm *.elf
	@rm *.txt
	@rm -rf ${IN_PATH}
	@rm -rf ${OUT_PATH}
