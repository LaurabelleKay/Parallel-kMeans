typedef struct point
{
    float x;
    float y;
    int cluster;
} point;

typedef struct centroid
{
    float oldx;
    float oldy;
    float meanx;
    float meany;
    float preMeanx;
    float preMeany;
    int count;
} centroid;

typedef long long hrtime_t;

void getDists(point p, centroid *C, int k, float *D);
float dist(point p, centroid c);
int nearest(float *D, int k);
void updateMean(int val, centroid *C, point p);
int stopCondition(centroid *C, int k);
void makeData(point *P, int n);
void writeFiles(centroid *C, point *P, int k, int n);
void readFiles(point *P, int n, char *x, char *y);
hrtime_t gethrtime();