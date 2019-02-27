all : Parallel

Parallel : kMeansCL.c
	nvcc kMeansCL.c -O3 -m64 -o kMeans -lOpenCL -I.
