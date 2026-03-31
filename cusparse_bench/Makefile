CC = nvcc
CUDA_INSTALL_PATH = $(CUDA_HOME)
NVCC_FLAGS = -O3 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
INCLUDES = -I$(CUDA_INSTALL_PATH)/include
LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse

all:
	$(CC) $(NVCC_FLAGS) src/cusparse_spgemm.cu -o cusparse_spgemm $(INCLUDES) $(LIBS)

clean:
	rm -f cusparse_spgemm
