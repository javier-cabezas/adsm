#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 16 * 1024 * 1024;

unsigned long vecSize = 0;
const size_t blockSize = 512;

const char *msg = "Done!";

const char *kernel = "\
__kernel void vecAdd( __global int *a, unsigned long size)\
{\
    int i = get_global_id(0);\
    if(i >= size) return;\
\
    a[i] = i;\
}\
";


int main(int argc, char *argv[])
{
	int *a;
	gmactime_t s, t;

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %lu\n", vecSize);

    getTime(&s);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(int)) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize = globalSize * localSize;
    assert(__oclConfigureCall(1, 0, &globalSize, &localSize) == gmacSuccess);
    assert(__oclPushArgument(gmacPtr(a)) == gmacSuccess);
    assert(__oclPushArgument(vecSize) == gmacSuccess);
    assert(__oclLaunch("vecAdd") == gmacSuccess);
    assert(gmacThreadSynchronize() == gmacSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float error = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        error += 1.0 * (a[i] - i);
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr, "Error: %f\n", error);

    gmacFree(a);

    return error != 0;
}
