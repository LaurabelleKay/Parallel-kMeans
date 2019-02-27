#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x10000)

#ifndef TILESIZE
#define TILESIZE 16
#endif
typedef long long hrtime_t;

hrtime_t gethrtime();
void initMatrix(int *A, int *B, int N);
void matMulDevice(int *A, int *B, int *C, int N);

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    int *A = malloc(N * N * sizeof(int));
    int *B = malloc(N * N * sizeof(int));
    int *C = malloc(N * N * sizeof(int));

    initMatrix(A, B, N);
    matMulDevice(A, B, C, N);
}

void initMatrix(int *A, int *B, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[N * i + j] = (int)rand() / (int)(RAND_MAX / 10);
            B[N * i + j] = (int)rand() / (int)(RAND_MAX / 10);
        }
    }
    return;
}

void matMulDevice(int *A, int *B, int *C, int N)
{
    FILE *file = fopen("./MatMul.cl", "r");
    if (!file)
    {
        printf("Failed to open Kernel\n");
        exit(1);
    }
    char *source = malloc(MAX_SOURCE_SIZE);
    size_t sourceSize = fread(source, 1, MAX_SOURCE_SIZE, file);
    fclose(file);

    //Platform info
    cl_platform_id platformID = NULL;
    cl_device_id deviceID = NULL;
    cl_uint numDevices;
    cl_uint NumPlatforms;

    cl_uint ret = clGetPlatformIDs(1, &platformID, &NumPlatforms);
    if (ret != CL_SUCCESS)
    {
        printf("GetPlatformID Error: %d\n", ret);
        exit(1);
    }
    ret = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &numDevices);
    if (ret != CL_SUCCESS)
    {
        printf("GetDeviceID Erro: %d\n", ret);
        exit(1);
    }

    //CL context
    cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateContext Error: %d\n", ret);
        exit(1);
    }

    //Create a command queue
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CommandQueue Error: %d\n", ret);
        exit(1);
    }

    //Matrix memory buffers
    cl_mem aMem = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateBuffer Error: %d\n", ret);
        exit(1);
    }
    cl_mem bMem = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateBuffer Error: %d\n", ret);
        exit(1);
    }
    cl_mem cMem = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateBuffer Error: %d\n", ret);
        exit(1);
    }
    cl_mem nMem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateBuffer Error: %d\n", ret);
        exit(1);
    }

    //Write to buffer object from host memory
    ret = clEnqueueWriteBuffer(commandQueue, aMem, CL_TRUE, 0, N * N * sizeof(int), A, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("WriteBuffer Error: %d\n", ret);
        exit(1);
    }
    ret = clEnqueueWriteBuffer(commandQueue, bMem, CL_TRUE, 0, N * N * sizeof(int), B, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("WriteBuffer Error: %d\n", ret);
        exit(1);
    }
    ret = clEnqueueWriteBuffer(commandQueue, nMem, CL_TRUE, 0, sizeof(int), &N, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("WriteBuffer Error: %d\n", ret);
        exit(1);
    }

    //create prgram from a kernel
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&sourceSize, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateProgram Error %d\n", ret);
        exit(1);
    }

    // Build the program
    ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("BuildProgram Error %d\n", ret);
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &ret);
    if (ret != CL_SUCCESS)
    {
        printf("CreateKernel Error %d\n", ret);
        exit(1);
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMem);
    if (ret != CL_SUCCESS)
    {
        printf("Error %d\n", ret);
        exit(1);
    }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bMem);
    if (ret != CL_SUCCESS)
    {
        printf("Error %d\n", ret);
        exit(1);
    }
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cMem);
    if (ret != CL_SUCCESS)
    {
        printf("Error %d\n", ret);
        exit(1);
    }
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&nMem);
    if (ret != CL_SUCCESS)
    {
        printf("Error %d\n", ret);
        exit(1);
    }

    size_t globalThreads[2] = {N, N};
    size_t localThreads[2] = {16, 16};

    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalThreads, localThreads, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("EnqueueKernel Error %d\n", ret);
        exit(1);
    }

    //TODO: Matrix multiply
    ret = clEnqueueReadBuffer(commandQueue, cMem, CL_TRUE, 0, N * N * sizeof(int), C, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("ReadBuffer Error %d\n", ret);
        exit(1);
    }

    FILE *outfile;
    outfile = fopen("output.txt", "w");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(outfile, "%d ", C[N * i + j]);
        }
        fprintf(outfile, "\n");
    }

    clReleaseMemObject(aMem);
    clReleaseMemObject(bMem);
    clReleaseMemObject(cMem);
}
