#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"


const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault =  16 * 1024 * 1024;
unsigned vecSize = 0;

const char *kernel = "\
__kernel void vecAdd(__global float *c, __global float *a, __global float *b, unsigned size)\
{\
    unsigned i = get_global_id(0);\
\
    c[i] = a[i] + b[i];\
}\
";


int main(int argc, char *argv[])
{
	float *a, *b, *c;
	gmactime_t s, t, S, T;

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    getTime(&s);
    assert(eclCompileSource(kernel) == eclSuccess);

    // Alloc & init input data
    assert(eclMalloc((void **)&a, vecSize * sizeof(float)) == eclSuccess);
    assert(eclMalloc((void **)&b, vecSize * sizeof(float)) == eclSuccess);
    // Alloc output data
    assert(eclMalloc((void **)&c, vecSize * sizeof(float)) == eclSuccess);
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    getTime(&S);
    getTime(&s);
    randInitMax(a, 10.f, vecSize);
    randInitMax(b, 10.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    float sum = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    getTime(&s);
    ecl_kernel kernel;
    size_t globalSize = vecSize;

    assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 0, c) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 1, a) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 2, b) == eclSuccess);
    assert(eclSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize) == eclSuccess);
    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");

    getTime(&s);
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        check += c[i];
    }
    getTime(&t);
    getTime(&T);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", fabsf(sum - check));
    printTime(&S, &T, "Total: ", "\n");

    eclReleaseKernel(kernel);

    eclFree(a);
    eclFree(b);
    eclFree(c);

   return sum != check;
}
