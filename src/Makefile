.DEFAULT_GOAL := compile

CC = ../compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
INPUT = main.cpp
OUTPUT = main_neon_cpp.elf


compile:
	@echo "Compiling to aarch64..."
	${CC} ${INPUT} -o ${OUTPUT}

run:
	@echo "Running compiled binary..."
	chmod +x ${OUTPUT} && ./${OUTPUT}

clean:
	@echo "Cleaning built binaries..."
	rm ${OUTPUT}