.DEFAULT_GOAL := aarch_64

aarch_64:
	@echo "Compiling to aarch64..."
	arm-linux-gnueabi-g++ -mfpu=neon-vfpv3 -o arm64main arm64main.cpp