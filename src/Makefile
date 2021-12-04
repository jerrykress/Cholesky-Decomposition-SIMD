# silent warnings
.DEFAULT_GOAL := compile
.SILENT:test

# color definitions
CYAN=\033[0;36m
NC=\033[0m
GREEN=\033[0;32m

# remove before submission
# ARM_CC = ~/compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++

# architecture option (=aarch64,x86_64)
TARGET_ARCH = x86_64
# arm compiler
ARM_CC = ../compiler/bin/aarch64-none-linux-gnu-g++
# intel compiler
X86_CC = g++
# compiler flags
# CC_VEC_FLAG = -O3
# available programs
PROG_SEQ  = main_seq
PROG_NEON = main_neon
PROG_AVX  = main_avx
PROG_NEON_OMP = main_neon_omp
PROG_AVX_OMP  = main_avx_omp
# [RUN] reference program
PROG_1 = ${PROG_SEQ}
# [RUN] benchmark program
PROG_2 = ${PROG_AVX_OMP}
# name of input and output folders
IN_PATH = input
OUT_PATH = output
# default input filename
TEST_FILE = input_${DIM}x${DIM}.txt
# compliation results from a testing session
RESULT_FILE = test_results.txt
# default test range
DIM1 ?= 32
DIM2 ?= 1024
# step increment during testing (default 1)
STEP ?= 1
# keep old input and output files from the last test (in case of testing failure, this servres as a checkpoint resume)
KEEP_HISTORY ?= 0
# generate new input files when testing, default true (turn off when inputs are preloaded, in case old input files from last test are still available)
GENERATE_NEW ?= 1
# floating point precision (=1,2) 
PRECISION ?= 1

install:
	@echo "${GREEN} Installing compiler... ${NC}"
	wget https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
	tar -Jxf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
	rm gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
	mv gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu ../compiler/
	make compile

compile:
	@echo "${GREEN} Target architecture: ${TARGET_ARCH} ${NC}"
	@echo "${GREEN} Compiling sequential ${NC}"
	@if [ ${TARGET_ARCH} = aarch64 ]; then\
        ${ARM_CC} ${CC_VEC_FLAG} ${PROG_1}.cpp -o ${PROG_1}.elf;\
    fi
	@if [ ${TARGET_ARCH} = x86_64 ]; then\
        ${X86_CC} ${CC_VEC_FLAG} ${PROG_1}.cpp -o ${PROG_1}.elf;\
    fi
	@echo "${GREEN} Compiling NEON single core ${NC}"
	${ARM_CC} ${CC_VEC_FLAG} -fopenmp -march=armv8-a+simd -mcpu=cortex-a72 -mtune=cortex-a72 ${PROG_NEON}.cpp -o ${PROG_NEON}.elf
	@echo "${GREEN} Compiling NEON multi core ${NC}"
	${ARM_CC} ${CC_VEC_FLAG} -fopenmp -march=armv8-a+simd -mcpu=cortex-a72 -mtune=cortex-a72 ${PROG_NEON_OMP}.cpp -o ${PROG_NEON_OMP}.elf
	@echo "${GREEN} Compiling AVX single core ${NC}"
	${X86_CC} ${CC_VEC_FLAG} -march=znver2 -mtune=znver2 -mavx ${PROG_AVX}.cpp -o ${PROG_AVX}.elf
	@echo "${GREEN} Compiling AVX multi core ${NC}"
	${X86_CC} ${CC_VEC_FLAG} -march=znver2 -mtune=znver2 -mavx -fopenmp ${PROG_AVX_OMP}.cpp -o ${PROG_AVX_OMP}.elf

generate:
	@echo "Generating random input size ${DIM}"
	@if [ ${GENERATE_NEW} = 1 ]; then\
        mkdir -p ${IN_PATH}; python3 generate_spd.py ${DIM}; mv ${TEST_FILE} ${IN_PATH};\
    fi
	@if [ ${GENERATE_NEW} = 0 ]; then\
    	echo "Attempting preloaded input. No new file generated";\
    fi

generate_batch:
	@echo "Generating batch input size ${DIM1} to ${DIM2}"
	@i=${DIM1}; while [ "$$i" -le ${DIM2} ]; do \
		clear; echo "${CYAN}Generating $$i / ${DIM2} ${NC}"; make -s generate DIM=$$i; i=$$((i + ${STEP}));\
  	done

run:
	@mkdir -p ${OUT_PATH}
	@echo "SEQ @ size=${DIM}"
	@chmod +x ${PROG_1}.elf && ./${PROG_1}.elf ${IN_PATH}/${TEST_FILE} ${PRECISION}
	@echo "SIMD @ size=${DIM}"
	@chmod +x ${PROG_2}.elf && ./${PROG_2}.elf ${IN_PATH}/${TEST_FILE} ${PRECISION}

test:
	@echo "Testing sizes from ${DIM1} up to ${DIM2}"
	@touch *
	@if [ ${KEEP_HISTORY} = 0 ]; then\
        rm -rf ${OUT_PATH}; rm -f ${RESULT_FILE};\
    fi
	@i=${DIM1}; while [ "$$i" -le ${DIM2} ]; do \
		clear; echo "${CYAN}Testing $$i / ${DIM2} ${NC}"; make -s generate DIM=$$i GENERATE_NEW=${GENERATE_NEW}; make -s run DIM=$$i; make -s check DIM=$$i; i=$$((i + ${STEP}));\
  	done
	@make -s graph
	@mv test_result.png ${DIM1}-${DIM2}-${STEP}.png
	@mv test_results.txt ${DIM1}-${DIM2}-${STEP}.txt

check:
	@echo "Compare output..."
	@python3 check_result.py ${DIM} ${OUT_PATH}/main_seq_out_${DIM}.txt ${OUT_PATH}/main_simd_out_${DIM}.txt ${OUT_PATH}/main_seq_perf_${DIM}.txt ${OUT_PATH}/main_simd_perf_${DIM}.txt

graph:
	@python3 graph.py

extract:
	@echo "${GREEN} Extracting sample inputs... ${NC}"
	tar -xzvf input.tar.gz

uninstall:
	@echo "Deleting compiler and cleaning up everything..."
	rm -rf ../compiler
	make clean

clean:
	@echo "Cleaning up everything..."
	@find ./ -maxdepth 1 -type f -name "*.elf" -delete 
	@find ./ -maxdepth 1 -type f -name "*.txt" -delete 
	@rm -rf ${IN_PATH}
	@rm -rf ${OUT_PATH}