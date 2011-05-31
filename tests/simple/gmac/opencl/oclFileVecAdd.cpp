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

float doTest(float *a, float *b, float *c, float *orig, const char *name)
{
    gmactime_t s, t;
    static char buffer[1024];

    FILE * fA = fopen(VECTORA, "rb");
    assert(fA != NULL);
    FILE * fB = fopen(VECTORB, "rb");
    assert(fB != NULL);
    getTime(&s);
    size_t ret = fread(a, sizeof(float), vecSize, fA);
    assert(ret == vecSize);
    ret = fread(b, sizeof(float), vecSize, fB);
    assert(ret == vecSize);

    fclose(fA);
    fclose(fB);
    getTime(&t);
    snprintf(buffer, 1024, "%s:Init: ", name);
    printTime(&s, &t, buffer, "\n");

    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    ocl_kernel kernel;

    assert(oclGetKernel("vecAdd", &kernel) == oclSuccess);

    cl_mem tmp = cl_mem(oclPtr(c));
    assert(oclSetKernelArg(kernel, 0, sizeof(cl_mem), &tmp) == oclSuccess);
    tmp = cl_mem(oclPtr(a));                              
    assert(oclSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp) == oclSuccess);
    tmp = cl_mem(oclPtr(b));                              
    assert(oclSetKernelArg(kernel, 2, sizeof(cl_mem), &tmp) == oclSuccess);
    assert(oclSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize) == oclSuccess);
    assert(oclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == oclSuccess);

    getTime(&t);
    snprintf(buffer, 1024, "%s:Run: ", name);
    printTime(&s, &t, buffer, "\n");

    getTime(&s);
    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += orig[i] - (c[i]);
    }
    getTime(&t);
    snprintf(buffer, 1024, "%s:Check: ", name);
    printTime(&s, &t, buffer, "\n");

    return error;
}

int main(int argc, char *argv[])
{
    float *a, *b, *c;
    gmactime_t s, t;
    float error_shared, error_distributed, error_centralized;

    assert(oclCompileSource(kernel) == oclSuccess);

    float * orig = (float *) malloc(vecSize * sizeof(float));
    FILE * fO = fopen(VECTORC, "rb");
    assert(fO != NULL);
    size_t ret = fread(orig, sizeof(float), vecSize, fO);
    assert(ret == vecSize);

    // Alloc output data
    assert(oclMalloc((void **)&c, vecSize * sizeof(float)) == oclSuccess);

    //////////////////////
    // Test shared objects
    //////////////////////
    getTime(&s);
    // Alloc & init input data
    assert(oclMalloc((void **)&a, vecSize * sizeof(float)) == oclSuccess);
    assert(oclMalloc((void **)&b, vecSize * sizeof(float)) == oclSuccess);
    getTime(&t);
    printTime(&s, &t, "Shared:Alloc: ", "\n");

    error_shared = doTest(a, b, c, orig, "Shared");

    getTime(&s);
    FILE * fC = fopen("vectorC_shared", "wb");
    assert(fC != NULL);
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "Shared:Write: ", "\n");

    getTime(&s);
    oclFree(a);
    oclFree(b);
    getTime(&t);
    printTime(&s, &t, "Shared:Free: ", "\n");

    //////////////////////////
    // Test replicated objects
    //////////////////////////
    getTime(&s);
    // Alloc & init input data
    assert(oclGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) == oclSuccess);
    assert(oclGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) == oclSuccess);
    getTime(&t);
    printTime(&s, &t, "Distributed:Alloc: ", "\n");

    error_distributed = doTest(a, b, c, orig, "Distributed");

    getTime(&s);
    fC = fopen("vectorC_replicated", "wb");
    ret = fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "Distributed:Write: ", "\n");

    getTime(&s);
    oclFree(a);
    oclFree(b);
    getTime(&t);
    printTime(&s, &t, "Distributed:Free: ", "\n");

    ///////////////////////////
    // Test centralized objects
    ///////////////////////////
    getTime(&s);
    // Alloc & init input data
    assert(oclGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) == oclSuccess);
    assert(oclGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) == oclSuccess);
    getTime(&t);
    printTime(&s, &t, "Centralized:Alloc: ", "\n");

    error_centralized = doTest(a, b, c, orig, "Centralized");

    getTime(&s);
    fC = fopen("vectorC_centralized", "wb");
    fwrite(c, sizeof(float), vecSize, fC);
    assert(ret == vecSize);
    fclose(fC);
    getTime(&t);
    printTime(&s, &t, "Centralized:Write: ", "\n");

    getTime(&s);
    oclFree(a);
    oclFree(b);
    getTime(&t);
    printTime(&s, &t, "Centralized:Write: ", "\n");

    oclFree(c);
    free(orig);
    return error_shared != 0.f && error_distributed != 0.f && error_centralized != 0.f;
}
