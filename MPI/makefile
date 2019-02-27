all : Parallel 
	
Parallel : kMeansMPI.c
	mpicc -g kMeansMPI.c -O3 -mtune=barcelona -march=barcelona -m64 -o kMeans -I.
		
