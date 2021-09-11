OPTIONS :=  -O3 --ptxas-options -v --gpu-architecture=sm_70 --compiler-bindir /usr/local/gcc-6.4.0/bin -std=c++11
CC = gcc
OBJS = timer.o
FLAGS = -fopenmp -O3
SEQFLAGS = -O3 

EXEC = knapSeq knapCUDA knapCUDAFull

all: $(EXEC)

knapKernel.o : knapKernel.h knapKernel.cu
	nvcc $(filter-out $<,$^) -c -o $@ $(OPTIONS)

knapCUDA : knapKernel.h knap.cu knapKernel.o timercu.o
	nvcc $(filter-out $<,$^) -o  $@ $(LIB) $(OPTIONS)
	
knapCUDAFull : knapKernel.h knap.cu knapKernel.o timercu.o
	nvcc $(filter-out $<,$^) -o  $@ $(LIB) $(OPTIONS) -DFULL

knapSeq: knap.c timer.o
	$(CC)  $(SEQFLAGS) -o knapSeq knap.c $(OBJS) 

timer.o: timer.c
	$(CC) -o $@ -c timer.c
	
timercu.o: timer.cu timercu.h
	nvcc $< -c -o $@ $(OPTIONS)

clean:
	rm -f $(EXEC) $(OBJS)

tar: Makefile sample timercu.o timercu.h timer.h timer.cu timer.c knapKernel.h knapKernel.cu knap.cu knap.c report.pdf
	tar cvf project_code.tar Makefile sample timercu.o timercu.h timer.h timer.cu timer.c knapKernel.h knapKernel.cu knap.cu knap.c report.pdf
