#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"


const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecAdd(float *c, float *a, float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    c[i] = a[i] + b[i];
}


float doTest(float *a, float *b, float *c, float *orig)
{
	struct timeval s, t;

    FILE * fA = fopen("inputset/vectorA", "r");
    FILE * fB = fopen("inputset/vectorB", "r");
    gettimeofday(&s, NULL);
    fread(a, sizeof(float), vecSize, fA);
    fread(b, sizeof(float), vecSize, fB);

    gettimeofday(&t, NULL);
    fclose(fA);
    fclose(fB);
    printTime(&s, &t, "Init: ", "\n");

    // Call the kernel
    gettimeofday(&s, NULL);
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), vecSize);
    if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Run: ", "\n");

    gettimeofday(&s, NULL);
    float error = 0;
    for(int i = 0; i < vecSize; i++) {
        error += orig[i] - (c[i]);
    }
    gettimeofday(&t, NULL);
    fprintf(stderr,"Error: %f\n", error);
    printTime(&s, &t, "Check: ", "\n");

    return error;
}

int main(int argc, char *argv[])
{
	float *a, *b, *c;
	struct timeval s, t;
    float error1, error2, error3;

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen("inputset/vectorC", "r");
    fread(orig, sizeof(float), vecSize, fO);

    // Alloc output data
    if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();

    //////////////////////
    // Test shared objects
    //////////////////////
    fprintf(stderr,"SHARED OBJECTS\n");
    gettimeofday(&s, NULL);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Alloc: ", "\n");

    error1 = doTest(a, b, c, orig);

    FILE * fC = fopen("vectorC_shared", "w");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    //////////////////////////
    // Test replicated objects
    //////////////////////////
    fprintf(stderr,"REPLICATED OBJECTS\n");
    gettimeofday(&s, NULL);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess)
        CUFATAL();
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Alloc: ", "\n");

    error2 = doTest(a, b, c, orig);

    fC = fopen("vectorC_replicated", "w");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    ///////////////////////////
    // Test centralized objects
    ///////////////////////////
    fprintf(stderr,"CENTRALIZED OBJECTS\n");
    gettimeofday(&s, NULL);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess)
        CUFATAL();
    gettimeofday(&t, NULL);
    printTime(&s, &t, "Alloc: ", "\n");

    error3 = doTest(a, b, c, orig);

    fC = fopen("vectorC_centralized", "w");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    gmacFree(c);
    free(orig);
    return error1 != 0.f && error2 != 0.f && error3 != 0.f;
}
