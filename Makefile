all:
	nvcc -O3 -o prog Matmul_COMPA.cu -Xcompiler -fopenmp -arch=sm_70