#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>

#include "utils.h"
#include "debug.h"

#ifdef _MSC_VER
#define VECTORA "inputset\\vectorA"
#define VECTORB "inputset\\vectorB"
#define VECTORC "inputset\\vectorC"
#else
#define VECTORA "inputset/vectorA"
#define VECTORB "inputset/vectorB"
#define VECTORC "inputset/vectorC"
#endif

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
    gmactime_t s, t;

    FILE * fA = fopen(VECTORA, "rb");
    FILE * fB = fopen(VECTORB, "rb");
    getTime(&s);
    size_t ret = fread(a, sizeof(float), vecSize, fA);
    assert(ret == vecSize);
    ret = fread(b, sizeof(float), vecSize, fB);
    assert(ret == vecSize);

    getTime(&t);
    fclose(fA);
    fclose(fB);
    printTime(&s, &t, "Init: ", "\n");

    // Call the kernel
    getTime(&s);
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;
    vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), vecSize);
    if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += orig[i] - (c[i]);
    }
    getTime(&t);
    fprintf(stderr, "Error: %f\n", error);
    printTime(&s, &t, "Check: ", "\n");

    return error;
}

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    gmactime_t s, t;
    float error1, error2, error3;

    fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen(VECTORC, "rb");
    size_t ret = fread(orig, sizeof(float), vecSize, fO);
    assert(ret == vecSize);

    // Alloc output data
    if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();

    //////////////////////
    // Test shared objects
    //////////////////////
    fprintf(stderr,"SHARED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error1 = doTest(a, b, c, orig);

    FILE * fC = fopen("vectorC_shared", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    //////////////////////////
    // Test replicated objects
    //////////////////////////
    fprintf(stderr,"REPLICATED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error2 = doTest(a, b, c, orig);

    fC = fopen("vectorC_replicated", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    ///////////////////////////
    // Test centralized objects
    ///////////////////////////
    fprintf(stderr,"CENTRALIZED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error3 = doTest(a, b, c, orig);

    fC = fopen("vectorC_centralized", "wb");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    gmacFree(c);
    free(orig);
    return error1 != 0.f && error2 != 0.f && error3 != 0.f;
}
