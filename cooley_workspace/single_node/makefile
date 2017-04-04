all: program

program: cppcode.o
	g++ main.cpp -std=c++11 -I/usr/local/cuda/include \
	-o convnet -L/usr/local/cuda/lib64 -lcuda \
	-L/usr/local/cuda/lib64 -lcudart -L/usr/local/cuda/lib64 -lcublas \
	-L/usr/local/cuda/lib64 -lcurand -L/mnt/d/code_libraries/cudnn/lib64 -lcudnn \
	FCLayerCU.o FCLayerCPP.o ConvLayerCU.o ConvLayerCPP.o
	rm *.o

cppcode.o: cudacode.o
	g++ -c -std=c++11 -L/usr/local/cuda/lib64 -lcuda -L/usr/local/cuda/lib64 \
	-lcudart -L/usr/local/cuda/lib64 -lcublas -L/usr/local/cuda/lib64 -lcurand \
	-L/usr/local/cuda/lib64 -lcudnn -o FCLayerCPP.o FCLayer.cpp
	g++ -c -std=c++11 -L/usr/local/cuda/lib64 -lcuda -L/usr/local/cuda/lib64 \
	-lcudart -L/usr/local/cuda/lib64 -lcublas -L/usr/local/cuda/lib64 -lcurand \
	-L/usr/local/cuda/lib64 -lcudnn -o ConvLayerCPP.o ConvLayer.cpp

cudacode.o:
	nvcc -c -std=c++11 -I/usr/local/cuda/include -o ConvLayerCU.o ConvLayer.cu \
	-gencode arch=compute_61,code=sm_61 -I/home/souham/downloads/cudnn
	nvcc -c -std=c++11 -I/usr/local/cuda/include -o FCLayerCU.o FCLayer.cu \
	-gencode arch=compute_61,code=sm_61 -I/home/souham/downloads/cudnn

clean:
	rm -rf *o convnet
