NVCC ?= nvcc

TARGET := rtss2026artifact_benchmark

.PHONY: all clean

all: $(TARGET)

$(TARGET): cuda_host_contention.cu
	$(NVCC) -O3 -std=c++17 -Xcompiler -pthread -lcuda -o $@ $<

clean:
	rm -f $(TARGET)
