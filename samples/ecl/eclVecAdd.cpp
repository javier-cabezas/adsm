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
    ecl_kernel kernel;
    assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
    cl_mem d_c = cl_mem(eclPtr(c));
    assert(eclSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c) == eclSuccess);
    cl_mem d_a = cl_mem(eclPtr(a));                        
    assert(eclSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a) == eclSuccess);
    cl_mem d_b = cl_mem(eclPtr(b));                        
    assert(eclSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b) == eclSuccess);
    assert(eclCallNDRange(kernel, 1, NULL, &vecSize, NULL) == eclSuccess);

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
