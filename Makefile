CC     ?= gcc
NVCC   ?= nvcc
CFLAGS  = -O3 -march=native -fopenmp -Wall
CUDA   ?= /usr/local/cuda

# Автовыбор: с nvcc собираем GPU-версию, иначе чистый CPU.
all:
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) --no-print-directory gpu; \
	else \
		echo "nvcc не найден — собираю CPU-версию"; \
		$(MAKE) --no-print-directory cpu; \
	fi

cpu: ox.c
	$(CC) $(CFLAGS) -fPIC -shared -o libox.so ox.c
	$(CC) $(CFLAGS) -o ox ox.c

gpu: ox.c ox_cuda.cu
	$(NVCC) -O3 -arch=native -c ox_cuda.cu -o ox_cuda.o
	$(CC) $(CFLAGS) -fPIC -shared -o libox.so ox.c
	$(CC) $(CFLAGS) -DOX_CUDA -c ox.c -o ox_main.o
	$(CC) $(CFLAGS) -o ox ox_main.o ox_cuda.o -L$(CUDA)/lib64 -lcudart -lstdc++

clean:
	rm -f ox libox.so ox_main.o ox_cuda.o

.PHONY: all cpu gpu clean
