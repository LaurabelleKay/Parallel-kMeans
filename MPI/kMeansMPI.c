#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "kMPI.h"

#pragma GCC diagnostic ignored "-Wunused-result"

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);

    int n;
    int k;

    char *xFilename;
    char *yFilename;
    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        n = 100;
    }
    printf("n: %d\n", n);
    if (argc >= 3)
    {
        k = atoi(argv[2]);
    }
    else
    {
        k = 4;
    }
    printf("k: %d\n", k);
    if (argc >= 5)
    {
        xFilename = argv[3];
        yFilename = argv[4];
    }
    else
    {
        xFilename = "Data/X100.txt";
        yFilename = "Data/Y100.txt";
    }

    int numPoints = n / nprocs;
    int offset = numPoints * rank;
    if (n % nprocs)
    {
        if (rank < n % nprocs)
        {
            numPoints++;
        }
        if (rank <= n % nprocs)
        {
            offset += rank;
        }
        else
        {
            offset += n % nprocs;
        }
    }

    point *points = malloc(numPoints * sizeof(point));
    point *rPoints = malloc(n * sizeof(point));

    //Create MPI Type for the centroids
    int nItems = 7;
    int blocklengths[7] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[7] = {
        MPI_FLOAT,
        MPI_FLOAT,
        MPI_FLOAT,
        MPI_FLOAT,
        MPI_FLOAT,
        MPI_FLOAT,
        MPI_INT,
    };

    MPI_Datatype centType;
    MPI_Aint offsets[7] = {
        offsetof(centroid, oldx),
        offsetof(centroid, oldy),
        offsetof(centroid, meanx),
        offsetof(centroid, meany),
        offsetof(centroid, preMeanx),
        offsetof(centroid, preMeany),
        offsetof(centroid, count),
    };
    MPI_Type_create_struct(nItems, blocklengths, offsets, types, &centType);
    MPI_Type_commit(&centType);

    //Create MPI type for the points
    nItems = 3;
    int bl[3] = {1, 1, 1};
    MPI_Datatype t[3] = {MPI_FLOAT, MPI_FLOAT, MPI_INT};
    MPI_Aint ofs[3] = {
        offsetof(point, x),
        offsetof(point, y),
        offsetof(point, cluster),
    };
    MPI_Datatype pointType;
    MPI_Type_create_struct(nItems, bl, ofs, t, &pointType);
    MPI_Type_commit(&pointType);

    MPI_Op reduce_op;
    MPI_Op_create((MPI_User_function *)reduceCentroids, 1, &reduce_op);

    centroid *centroids = malloc(k * sizeof(centroid));
    centroid *rCentroids = malloc(k * sizeof(centroid));

    for (int i = 0; i < k; i++)
    {
        centroids[i].count = 0;
        centroids[i].preMeanx = 0.0;
        centroids[i].preMeany = 0.0;
        centroids[i].meanx = 0.0;
        centroids[i].meany = 0.0;
        if (rank == 0)
        {
            float range = 8;
            centroids[i].meanx = range * ((float)rand() / (float)RAND_MAX);
            centroids[i].meany = range * ((float)rand() / (float)RAND_MAX);
        }
        centroids[i].oldx = 0;
        centroids[i].oldy = 0;
    }
    MPI_Bcast(centroids, k, centType, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        readFiles(rPoints, n, xFilename, yFilename);
    }

    hrtime_t start, end;
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        start = gethrtime();
    }

    int *disps = malloc(nprocs * sizeof(int));
    MPI_Gather(&offset, 1, MPI_INT, disps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *recvcounts = malloc(nprocs * sizeof(int));
    MPI_Gather(&numPoints, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rPoints, recvcounts, disps, pointType, points, numPoints, pointType, 0, MPI_COMM_WORLD);

    int ret;

    float *distances = malloc(k * sizeof(float));
    int iters = 0;

    do
    {
        iters++;
        for (int i = 0; i < numPoints; i++)
        {
            getDists(points[i], centroids, k, distances);
            points[i].cluster = nearest(distances, k);
            updateMean(points[i].cluster, centroids, points[i]);
        }
        MPI_Reduce(centroids, rCentroids, k, centType, reduce_op, 0, MPI_COMM_WORLD);
        if (rank == 0) //Update the means
        {
            for (int i = 0; i < k; i++)
            {

                if (rCentroids[i].count == 0)
                {
                    rCentroids[i].preMeanx = 0;
                    rCentroids[i].preMeany = 0;
                    continue;
                }
                centroids[i].meanx = rCentroids[i].preMeanx / rCentroids[i].count;
                centroids[i].meany = rCentroids[i].preMeany / rCentroids[i].count;
                rCentroids[i].count = 0;
                rCentroids[i].preMeanx = 0;
                rCentroids[i].preMeany = 0;
            }
        }
        MPI_Bcast(centroids, k, centType, 0, MPI_COMM_WORLD);
    } while (!stopCondition(centroids, k));

    MPI_Gatherv(points, numPoints, pointType, rPoints, recvcounts, disps, pointType, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        end = gethrtime();
        float tim = (end - start) / 1E09;
        printf("total seconds: %f\n", tim);

    }

     if (rank == 0)
    {
        writeFiles(centroids, rPoints, k, n);
    }

    MPI_Finalize();
    return (0);
}

void readFiles(point *P, int n, char *x, char *y)
{
    FILE *xFile = fopen(x, "r");
    FILE *yFile = fopen(y, "r");
    if (xFile == NULL || yFile == NULL)
    {
        printf("Could not read files\n");
        exit(1);
    }
    for (int i = 0; i < n; i++)
    {
        fscanf(xFile, "%f", &P[i].x);
        fscanf(yFile, "%f", &P[i].y);
    }
    fclose(xFile);
    fclose(yFile);
}

//Returns the distance between 2 points
float dist(point p, centroid c)
{
    float x = p.x - c.meanx;
    float y = p.y - c.meany;
    float distance = x * x + y * y;
    return distance;
}

//Find the distances from a point to all centroids
void getDists(point p, centroid *C, int k, float *D)
{
    for (int i = 0; i < k; i++)
    {
        D[i] = dist(p, C[i]);
    }
}

//Returns the index of the nearest centroid to a point
int nearest(float *D, int k)
{
    float min_dist = 1E06;
    int min_i;
    for (int i = 0; i < k; i++)
    {
        if (D[i] < min_dist)
        {
            min_dist = D[i];
            min_i = i;
        }
    }
    return min_i;
}

//Add to the mean for a centroid
void updateMean(int val, centroid *C, point p)
{
    C[val].count++;
    C[val].preMeanx += p.x;
    C[val].preMeany += p.y;
}

//Checks if the new mean of all centroids differ from the old values
int stopCondition(centroid *C, int k)
{
    int ret = 1;
    double diffx;
    double diffy;
    for (int i = 0; i < k; i++)
    {
        diffx = fabs(C[i].oldx - C[i].meanx);
        diffy = fabs(C[i].oldy - C[i].meany);

        //If centroid has no points assigned to it
        if (C[i].count == 0)
        {
            continue;
        }
        if (diffx > 0 || diffy > 0)
        {
            ret = 0;
        }

        C[i].oldx = C[i].meanx;
        C[i].oldy = C[i].meany;

        C[i].preMeanx = 0.0;
        C[i].preMeany = 0.0;
        C[i].count = 0;
    }
    return ret;
}

//Reduce operation for the centorid struct
void reduceCentroids(void *in, void *out, int *len, MPI_Datatype *typeptr)
{
    centroid *inv = in;
    centroid *outv = out;
    for (int i = 0; i < *len; i++)
    {

        outv[i].preMeanx += inv[i].preMeanx;
        outv[i].preMeany += inv[i].preMeany;
        outv[i].count += inv[i].count;
    }
}

void writeFiles(centroid *C, point *P, int k, int n)
{
    FILE *cFile = fopen("../MPIC.txt", "w");
    for (int i = 0; i < n; i++)
    {
        fprintf(cFile, "%d", P[i].cluster);
        if (i < n - 1)
        {
            fprintf(cFile, "\n");
        }
    }

    FILE *cxFile = fopen("../MPICX.txt", "w");
    for (int i = 0; i < k; i++)
    {
        fprintf(cxFile, "%f", C[i].meanx);
        if (i < k - 1)
        {
            fprintf(cxFile, "\n");
        }
    }

    FILE *cyFile = fopen("../MPICY.txt", "w");
    for (int i = 0; i < k; i++)
    {
        fprintf(cyFile, "%f", C[i].meany);
        if (i < k - 1)
        {
            fprintf(cyFile, "\n");
        }
    }

    fclose(cFile);
    fclose(cyFile);
    fclose(cxFile);
}

hrtime_t gethrtime()
{
    hrtime_t elapsed;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);          //multicore - safe, but includes outside interference
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); // maynot be multicore -safe if process migrates cores
    elapsed = (hrtime_t)t.tv_sec * (hrtime_t)1e9 + (hrtime_t)t.tv_nsec;
    return elapsed;
}