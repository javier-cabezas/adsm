#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <gmac/opencl.h>

unsigned vecSize = 16 * 1024 * 1024;

const char *kernel = "\
__kernel void vecAdd(__global float *c, __global float *a, __global float *b)\
{\
    unsigned i = get_global_id(0);\
\
    c[i] = a[i] + b[i];\
}\
";


int main(int argc, char *argv[])
{
	float *a, *b, *c;

    assert(eclCompileSource(kernel) == eclSuccess);

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    // Alloc & init input data
    assert(eclMalloc((void **)&a, vecSize * sizeof(float)) == eclSuccess);
    assert(eclMalloc((void **)&b, vecSize * sizeof(float)) == eclSuccess);
    // Alloc output data
    assert(eclMalloc((void **)&c, vecSize * sizeof(float)) == eclSuccess);

    for(unsigned i = 0; i < vecSize; i++) {
        a[i] = 1.f * rand() / RAND_MAX;
        b[i] = 1.f * rand() / RAND_MAX;
    }

    // Call the kernel
    size_t globalSize = vecSize;
    ecl_kernel kernel;
    assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 0, c) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 1, a) == eclSuccess);
    assert(eclSetKernelArgPtr(kernel, 2, b) == eclSuccess);
    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

    // Check the result in the CPU
    float error = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
    }
    fprintf(stderr, "Error: %f\n", error);

    eclReleaseKernel(kernel);

    eclFree(a);
    eclFree(b);
    eclFree(c);

   return 0;
}
