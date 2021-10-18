.DEFAULT_GOAL := compile

compile:
	@echo "Compiling to aarch64..."
	../compiler/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ main.cpp -o main_neon_cpp.elf

run:
	@echo "Running compiled binary..."
	chmod +x main_neon_cpp.elf && ./main_neon_cpp.elf