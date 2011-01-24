#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

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

const char *kernel = "\
__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned long size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    c[i] = a[i] + b[i];\
}\
";

float doTest(float *a, float *b, float *c, float *orig)
{
    gmactime_t s, t;

    FILE * fA = fopen(VECTORA, "rb");
    FILE * fB = fopen(VECTORB, "rb");
    getTime(&s);
    int ret = fread(a, sizeof(float), vecSize, fA);
    ret = fread(b, sizeof(float), vecSize, fB);

    getTime(&t);
    fclose(fA);
    fclose(fB);
    printTime(&s, &t, "Init: ", "\n");

    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;
    assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
    cl_mem tmp = cl_mem(gmacPtr(c));
    __oclPushArgument(&tmp, sizeof(cl_mem));
    tmp = cl_mem(gmacPtr(a));
    __oclPushArgument(&tmp, sizeof(cl_mem));
    tmp = cl_mem(gmacPtr(b));
    __oclPushArgument(&tmp, sizeof(cl_mem));
    __oclPushArgument(&vecSize, sizeof(vecSize));
    assert(__oclLaunch("vecAdd") == gmacSuccess);
    assert(gmacThreadSynchronize() == gmacSuccess);

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

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen(VECTORC, "rb");
    int ret = fread(orig, sizeof(float), vecSize, fO);

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
    fwrite(c, sizeof(float), vecSize, fC);
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

    fC = fopen("vectorC_replicated", "w");
    fwrite(c, sizeof(float), vecSize, fC);
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

    fC = fopen("vectorC_centralized", "w");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    gmacFree(c);
    free(orig);
    return error1 != 0.f && error2 != 0.f && error3 != 0.f;
}
