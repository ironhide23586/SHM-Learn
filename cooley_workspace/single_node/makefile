all: program

program: cppcode.o
	g++ -std=c++11 -o convnet -L/home/souham/.local/cuda-8.0/lib64 -lcuda \
	-lcudart -lcublas -lcurand -L/home/souham/downloads/cudnn -lcudnn main.cpp FCLayerCU.o FCLayerCPP.o \
	ConvLayerCU.o ConvLayerCPP.o
	rm *.o

cppcode.o: cudacode.o
	g++ -c -std=c++11 -I/home/souham/downloads/cudnn -o FCLayerCPP.o FCLayer.cpp
	g++ -c -std=c++11 -I/home/souham/downloads/cudnn -o ConvLayerCPP.o ConvLayer.cpp

cudacode.o:
	nvcc -c -std=c++11 -o ConvLayerCU.o ConvLayer.cu
	nvcc -c -std=c++11 -o FCLayerCU.o FCLayer.cu 

clean:
	rm -rf *o convnet
