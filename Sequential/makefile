all : Sequential
	
Sequential : kMeans.c
	gcc kMeans.c -O3 -mtune=barcelona -march=barcelona -m64 -o kMeans -I.