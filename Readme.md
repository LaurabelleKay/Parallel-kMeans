# Parallel Implementations of the k-Means Algorithm

Sequential, MPI, and OpenCL implementations of the k-means clustering algorithm.

## Prerequisites
* [OpenMPI](https://www.open-mpi.org/)
* OpenCL
* Python 2.7 (or greater)

## Input format
A text file for the x coordinates and another for the y, example inputs are in the Data folder

## Run instructions
Compile the programs with 

```
make
```
Note: The default OpenCL compiler is nvcc, edit the makefile for AMD/Intel GPUs

### Sequential
```
./kMeans [n] [k] [x file] [y file]
```
Example: 
```
./kMeans 1000 8 Data/X1000.txt Data/Y1000.txt
```
### MPI
```
./kMeanMPI [n] [k] [x file] [y file]
```

### OpenCL
```
./kMeansCL [n] [k] [x file] [y file]
```
## Visualisation 

### Sequential
```
python plot.py [x file] [y file]
```
### MPI
```
python plotMPI.py [x file] [y file]
```

### OpenCL
```
python plotOCL.py [x file] [y file]
```