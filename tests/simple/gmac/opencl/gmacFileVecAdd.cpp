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

const unsigned vecSize = 1024 * 1024;
const unsigned blockSize = 32;

const char *msg = "Done!";

const char *kernel = "\
__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
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
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    ocl_kernel kernel;

    assert(__oclKernelGet("vecAdd", &kernel) == oclSuccess);

    assert(__oclKernelConfigure(&kernel, 1, NULL, &globalSize, &localSize) == oclSuccess);
    cl_mem tmp = cl_mem(oclPtr(c));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 0) == oclSuccess);
    tmp = cl_mem(oclPtr(a));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 1) == oclSuccess);
    tmp = cl_mem(oclPtr(b));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 2) == oclSuccess);
    assert(__oclKernelSetArg(&kernel, &vecSize, sizeof(vecSize), 3) == oclSuccess);
    assert(__oclKernelLaunch(&kernel) == oclSuccess);

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

    assert(__oclPrepareCLCode(kernel) == oclSuccess);

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen(VECTORC, "rb");
    size_t ret = fread(orig, sizeof(float), vecSize, fO);
    assert(ret == vecSize);

    // Alloc output data
    if(oclMalloc((void **)&c, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();

    //////////////////////
    // Test shared objects
    //////////////////////
    fprintf(stderr,"SHARED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(oclMalloc((void **)&a, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    if(oclMalloc((void **)&b, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error1 = doTest(a, b, c, orig);

    FILE * fC = fopen("vectorC_shared", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);

    fclose(fC);

    oclFree(a);
    oclFree(b);

    //////////////////////////
    // Test replicated objects
    //////////////////////////
    fprintf(stderr,"REPLICATED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != oclSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != oclSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error2 = doTest(a, b, c, orig);

    fC = fopen("vectorC_replicated", "wb");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    oclFree(a);
    oclFree(b);

    ///////////////////////////
    // Test centralized objects
    ///////////////////////////
    fprintf(stderr,"CENTRALIZED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != oclSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != oclSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error3 = doTest(a, b, c, orig);

    fC = fopen("vectorC_centralized", "wb");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    oclFree(a);
    oclFree(b);

    oclFree(c);
    free(orig);
    return error1 != 0.f && error2 != 0.f && error3 != 0.f;
}
