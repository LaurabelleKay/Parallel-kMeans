all : Sequential MPI OpenCL

Sequential :  Sequential/kMeans.c
	gcc Sequential/kMeans.c -O3 -march=native -o kMeans -I.

OpenCL : OpenCL/kMeansCL.c
	nvcc OpenCL/MeansCL.c -O3 -m64 -o kMeans -lOpenCL -I.

MPI : MPI/kMeansMPI.c
	mpicc MPI/kMeansMPI.c -O3 -march=native -o kMeansMPI -I.
