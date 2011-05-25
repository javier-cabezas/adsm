#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 16 * 1024 * 1024;

unsigned vecSize = 0;
const unsigned blockSize = 512;

const char *msg = "Done!";

const char *kernel = "\
__kernel void vecAdd( __global int *a, unsigned size)\
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

    assert(oclCompileSource(kernel) == oclSuccess);

    setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

    getTime(&s);
    // Alloc & init input data
    if(oclMalloc((void **)&a, vecSize * sizeof(int)) != oclSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize = globalSize * localSize;
    ocl_kernel kernel;

    assert(oclGetKernel("vecAdd", &kernel) == oclSuccess);

    cl_mem tmp = cl_mem(oclPtr(a));
    assert(oclSetKernelArg(kernel, 0, &tmp, sizeof(cl_mem)) == oclSuccess);
    assert(oclSetKernelArg(kernel, 1, &vecSize, sizeof(vecSize)) == oclSuccess);

    assert(oclCallNDRange(kernel, 1, 0, &globalSize, &localSize) == oclSuccess);

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
