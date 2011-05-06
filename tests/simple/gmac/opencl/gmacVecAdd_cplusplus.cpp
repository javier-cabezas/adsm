#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl>

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

    assert(ocl::__prepareCLCode(kernel) == oclSuccess);

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    // Alloc & init input data
    if(ocl::malloc((void **)&a, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    if(ocl::malloc((void **)&b, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    // Alloc output data
    if(ocl::malloc((void **)&c, vecSize * sizeof(float)) != oclSuccess)
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

    ocl::KernelLaunch kernel;
    assert(kernel.get("vecAdd") == oclSuccess);
    assert(kernel.configure(1, NULL, &globalSize, &localSize) == oclSuccess);
    assert(kernel.setArg(c, 0) == oclSuccess);
    assert(kernel.setArg(a, 1) == oclSuccess);
    assert(kernel.setArg(b, 2) == oclSuccess);
    assert(kernel.setArg(vecSize, 3) == oclSuccess);
    assert(kernel.launch() == oclSuccess);

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

    ocl::free(a);
    ocl::free(b);
    ocl::free(c);

    //return error != 0;
    return 0;
}
