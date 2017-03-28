all: program

program: cppcode.o
	g++ -std=c++11 -o convnet -L/soft/visualization/cuda-8.0.61/lib64 -lcuda \
	-lcudart -lcudnn -lcublas -lcurand main.cpp FCLayerCU.o FCLayerCPP.o \
	ConvLayerCU.o ConvLayerCPP.o -nostartfiles
	rm *.o

cppcode.o: cudacode.o
	g++ -c -std=c++11 -o FCLayerCPP.o FCLayer.cpp
	g++ -c -std=c++11 -o ConvLayerCPP.o ConvLayer.cpp

cudacode.o:
	nvcc -c -std=c++11 -arch=sm_37 -o ConvLayerCU.o ConvLayer.cu
	nvcc -c -std=c++11 -arch=sm_37 -o FCLayerCU.o FCLayer.cu 

clean:
	rm -rf *o convnet