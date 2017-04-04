CUDA_PATH=/home/souham/.local/cuda-8.0

all: program

program: cppcode.o
	g++ main.cpp -std=c++11 -I$(CUDA_PATH)/include \
	-o convnet -L$(CUDA_PATH)/lib64 -lcuda \
	-L$(CUDA_PATH)/lib64 -lcudart -L$(CUDA_PATH)/lib64 -lcublas \
	-L$(CUDA_PATH)/lib64 -lcurand -L$(CUDA_PATH)/lib64 -lcudnn \
	FCLayerCU.o FCLayerCPP.o ConvLayerCU.o ConvLayerCPP.o
	rm *.o

cppcode.o: cudacode.o
	g++ -c -std=c++11 -L$(CUDA_PATH)/lib64 -lcuda -L$(CUDA_PATH)/lib64 \
	-lcudart -L$(CUDA_PATH)/lib64 -lcublas -L$(CUDA_PATH)/lib64 -lcurand \
	-L$(CUDA_PATH)/lib64 -lcudnn -o FCLayerCPP.o FCLayer.cpp
	g++ -c -std=c++11 -L$(CUDA_PATH)/lib64 -lcuda -L$(CUDA_PATH)/lib64 \
	-lcudart -L$(CUDA_PATH)/lib64 -lcublas -L$(CUDA_PATH)/lib64 -lcurand \
	-L$(CUDA_PATH)/lib64 -lcudnn -o ConvLayerCPP.o ConvLayer.cpp

cudacode.o:
	nvcc -c -std=c++11 -I$(CUDA_PATH)/include -o ConvLayerCU.o ConvLayer.cu \
	-gencode arch=compute_37,code=sm_37 -I$(CUDA_PATH)/include
	nvcc -c -std=c++11 -I$(CUDA_PATH)/include -o FCLayerCU.o FCLayer.cu \
	-gencode arch=compute_37,code=sm_37 -I$(CUDA_PATH)/include

clean:
	rm -rf *o convnet