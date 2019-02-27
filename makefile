all : Sequential MPI OpenCL

Sequential :  Sequential/kMeans.c
	gcc Sequential/kMeans.c -O3 -mtune=barcelona -march=barcelona -m64 -o kMeans -I.

OpenCL : OpenCL/kMeansCL.c
	nvcc OpenCL/MeansCL.c -O3 -m64 -o kMeans -lOpenCL -I.

MPI : MPI/kMeansMPI.c
	mpicc -g MPI/kMeansMPI.c -O3 -mtune=barcelona -march=barcelona -m64 -o kMeansMPI -I.