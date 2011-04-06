#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 16 * 1024 * 1024;

size_t vecSize = 0;
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

    getTime(&s);
    // Alloc & init input data
    if(oclMalloc((void **)&a, vecSize * sizeof(int)) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize = globalSize * localSize;
    OclKernel kernel;

    assert(__oclKernelGet("vecAdd", &kernel) == gmacSuccess);

    assert(__oclKernelConfigure(&kernel, 1, 0, &globalSize, &localSize) == gmacSuccess);
    cl_mem tmp = cl_mem(oclPtr(a));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 0) == gmacSuccess);
    assert(__oclKernelSetArg(&kernel, &vecSize, 8, 1) == gmacSuccess);
    assert(__oclKernelLaunch(&kernel) == gmacSuccess);
    assert(oclThreadSynchronize() == gmacSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float error = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        error += 1.0f * (a[i] - i);
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr, "Error: %f\n", error);

    oclFree(a);

    return error != 0;
}
