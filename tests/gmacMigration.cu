#include <gmac.h>

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 1 * 1024 * 1024;

size_t vecSize = 0;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecInc(float *a, size_t size)
{
    int i =  threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    a[i] += 1;
}

#define ITER 250

int main(int argc, char *argv[])
{
    float *a = NULL;
    struct timeval s, t;

    setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
    fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    gmacMigrate(0);
    gettimeofday(&s, NULL);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    gmacMemset(a, 0, vecSize * sizeof(float));
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Alloc: ", "\n");
    
    // Call the kernel
    gettimeofday(&s, NULL);
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;

    for(int i = 0; i < ITER; i++) {
        vecInc<<<Dg, Db>>>(gmacPtr(a), vecSize);
        if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    }
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Run no-migrate: ", "\n");

    gmacMemset(a, 0, vecSize * sizeof(float));
    gettimeofday(&s, NULL);
    for(int i = 0; i < ITER; i++) {
        vecInc<<<Dg, Db>>>(gmacPtr(a), vecSize);
        if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
        gmacMigrate(i % 2);
    }
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Run migrate: ", "\n");

    float error = 0;
    for(int i = 0; i < vecSize; i++) {
        error += a[i] - float(2*ITER);
    }
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr,"Error: %f\n", error);

    gmacFree(a);

    return error != 0.f;
}
