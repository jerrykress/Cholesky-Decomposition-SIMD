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

# default input filename
TEST_FILE = input_${DIM}x${DIM}.txt
# compliation results from a testing session
RESULT_FILE = test_results.txt
# step increment during testing (default 1)
STEP ?= 1
# keep old input and output files from the last test (in case of testing failure, this servres as a checkpoint resume)
KEEP_HISTORY ?= 0
# generate new input files when testing, default true (turn off when inputs are preloaded, in case old input files from last test are still available)
GENERATE_NEW ?= 1
# Test mode(1,2) 
# 1 - float32, 2 - float64
TEST_MODE ?= 1

generate:
	@echo "Generating random input size ${DIM}"
	@if [ ${GENERATE_NEW} = 1 ]; then\
        mkdir -p ${IN_PATH}; python3 generate_spd.py ${DIM}; mv ${TEST_FILE} ${IN_PATH};\
    fi
	@if [ ${GENERATE_NEW} = 0 ]; then\
    	echo "Attempting preloaded input. No new file generated";\
    fi

compile:
	@echo "Compiling sequential code..."
	${CC} ${PROG_SEQ}.cpp -o ${PROG_SEQ}.elf
	@echo "Compiling SIMD code..."
	${CC} -march=armv8-a+simd -mcpu=cortex-a72 -mtune=cortex-a72 ${PROG_SIMD}.cpp -o ${PROG_SIMD}.elf

run:
	@mkdir -p ${OUT_PATH}
	@echo "SEQ @ size=${DIM}"
	@chmod +x ${PROG_SEQ}.elf && ./${PROG_SEQ}.elf ${IN_PATH}/${TEST_FILE} ${TEST_MODE}
	@echo "SIMD @ size=${DIM}"
	@chmod +x ${PROG_SIMD}.elf && ./${PROG_SIMD}.elf ${IN_PATH}/${TEST_FILE} ${TEST_MODE}

test:
	@echo "Testing sizes from ${DIM1} up to ${DIM2}"
	@touch *
	@if [ ${KEEP_HISTORY} = 0 ]; then\
        rm -rf ${IN_PATH}; rm -rf ${OUT_PATH}; rm -f ${RESULT_FILE};\
    fi
	@i=${DIM1}; while [ "$$i" -le ${DIM2} ]; do \
		clear; echo "${CYAN}Testing $$i / ${DIM2} ${NC}"; make -s generate DIM=$$i GENERATE_NEW=${GENERATE_NEW}; make -s run DIM=$$i; make -s check DIM=$$i; i=$$((i * ${STEP}));\
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
