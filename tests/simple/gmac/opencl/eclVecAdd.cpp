#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 16 * 1024 * 1024;
unsigned vecSize = 0;

const size_t blockSize = 256;

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

    assert(eclCompileSource(kernel) == eclSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    // Alloc & init input data
    if(eclMalloc((void **)&a, vecSize * sizeof(float)) != eclSuccess)
        CUFATAL();
    if(eclMalloc((void **)&b, vecSize * sizeof(float)) != eclSuccess)
        CUFATAL();
    // Alloc output data
    if(eclMalloc((void **)&c, vecSize * sizeof(float)) != eclSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    float sum = 0.f;

    getTime(&s);
    randInit(a, vecSize);
    randInit(b, vecSize);
    //init(a, int(vecSize), 1.f);
    //init(b, int(vecSize), 1.f);
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

    ecl_kernel kernel;

    assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
    cl_mem tmp = cl_mem(eclPtr(c));
    assert(eclSetKernelArg(kernel, 0, sizeof(cl_mem), &tmp) == eclSuccess);
    tmp = cl_mem(eclPtr(a));                        
    assert(eclSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp) == eclSuccess);
    tmp = cl_mem(eclPtr(b));                        
    assert(eclSetKernelArg(kernel, 2, sizeof(cl_mem), &tmp) == eclSuccess);
    assert(eclSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize) == eclSuccess);
    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == eclSuccess);

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

    eclReleaseKernel(kernel);

    eclFree(a);
    eclFree(b);
    eclFree(c);

    //return error != 0;
    return 0;
}
