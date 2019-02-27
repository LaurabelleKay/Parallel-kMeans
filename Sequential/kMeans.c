#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../kMeans.h"

#pragma GCC diagnostic ignored "-Wunused-result"

int main(int argc, char **argv)
{
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
    point *points = malloc(n * sizeof(point));
    readFiles(points, n, xFilename, yFilename);
    centroid *centroids = malloc(k * sizeof(centroid));
    for (int i = 0; i < k; i++)
    {
        centroids[i].count = 0;
        centroids[i].preMeanx = 0.0;
        centroids[i].preMeany = 0.0;
        float range = 8;
        centroids[i].meanx = range * ((float)rand() / (float)RAND_MAX);
        centroids[i].meany = range * ((float)rand() / (float)RAND_MAX);
    }

    float *distances = malloc(k * sizeof(float));
    int cluster;
    int iters = 0;
    hrtime_t start = gethrtime();
    
    do
    {
        iters++;
        for (int i = 0; i < n; i++)
        {
            getDists(points[i], centroids, k, distances);
            points[i].cluster = nearest(distances, k);
            updateMean(points[i].cluster, centroids, points[i]);
        }
        for (int i = 0; i < k; i++)
        {
            if (centroids[i].count == 0)
            {
                continue;
            }
            centroids[i].meanx = centroids[i].preMeanx / centroids[i].count;
            centroids[i].meany = centroids[i].preMeany / centroids[i].count;
        }
    } while (!stopCondition(centroids, k));

    hrtime_t end = gethrtime();
    float t = (end - start) / 1E09;

    printf("total seconds: %f\n", t);

    writeFiles(centroids, points, k, n);
    return (0);
}

void readFiles(point *P, int n, char *x, char *y)
{
    int ret;
    FILE *xFile = fopen(x, "r");
    FILE *yFile = fopen(y, "r");
    printf("%s\n",x);
    printf("%s\n", y);
    if (xFile == NULL || yFile == NULL)
    {
        printf("Could not read files\n");
        exit(1);
    }
    for (int i = 0; i < n; i++)
    {
        fscanf(xFile, "%f", &P[i].x);
        fscanf(yFile, "%f", &P[i].y);
        //printf("%d\n", i);
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

//Update the means for a centroid
void updateMean(int val, centroid *C, point p)
{
    C[val].count++;
    C[val].preMeanx += p.x;
    C[val].preMeany += p.y;
}

int stopCondition(centroid *C, int k)
{
    int ret = 1;
    double diffx;
    double diffy;
    for (int i = 0; i < k; i++)
    {
        diffx = fabs(C[i].oldx - C[i].meanx);
        diffy = fabs(C[i].oldy - C[i].meany);

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

void writeFiles(centroid *C, point *P, int k, int n)
{
    FILE *cFile = fopen("C.txt", "w");
    for (int i = 0; i < n; i++)
    {
        fprintf(cFile, "%d", P[i].cluster);
        if (i < n - 1)
        {
            fprintf(cFile, "\n");
        }
    }

    FILE *cxFile = fopen("CX.txt", "w");
    for (int i = 0; i < k; i++)
    {
        fprintf(cxFile, "%f", C[i].meanx);
        if (i < k - 1)
        {
            fprintf(cxFile, "\n");
        }
    }

    FILE *cyFile = fopen("CY.txt", "w");
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