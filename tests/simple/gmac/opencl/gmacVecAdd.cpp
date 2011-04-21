#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 32 * 1024 * 1024;
unsigned vecSize = 0;

const size_t blockSize = 32;

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


int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t;

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    // Alloc & init input data
    if(oclMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(oclMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    // Alloc output data
    if(oclMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    float sum = 0.f;

    getTime(&s);
    valueInit(a, 1.f, vecSize);
    valueInit(b, 1.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    getTime(&s);
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    OclKernel kernel;

    assert(__oclKernelGet("vecAdd", &kernel) == gmacSuccess);
    assert(__oclKernelConfigure(&kernel, 1, NULL, &globalSize, &localSize) == gmacSuccess);
    cl_mem tmp = cl_mem(oclPtr(c));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 0) == gmacSuccess);
    tmp = cl_mem(oclPtr(a));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 1) == gmacSuccess);
    tmp = cl_mem(oclPtr(b));
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 2) == gmacSuccess);
    assert(__oclKernelSetArg(&kernel, &vecSize, sizeof(vecSize), 3) == gmacSuccess);
    assert(__oclKernelLaunch(&kernel) == gmacSuccess);
    assert(__oclKernelWait(&kernel) == gmacSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float error = 0.f;
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
        check += c[i];
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", error);

    if (sum != check) {
        printf("Sum: %f vs %f\n", sum, check);
        abort();
    }

    oclFree(a);
    oclFree(b);
    oclFree(c);

    //return error != 0;
    return 0;
}
