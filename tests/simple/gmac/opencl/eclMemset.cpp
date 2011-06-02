#include <stdio.h>
#include <cstring>
#include <gmac/opencl.h>

const size_t size = 4 * 1024 * 1024;
const size_t blockSize = 512;

const char *kernel = "\
__kernel void reset(__global long *a, unsigned long size, long v)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
	a[i] += v;\
}\
";

int check(long *ptr, int s)
{
	int a = 0;
	for(unsigned i = 0; i < size; i++)
		a += ptr[i];
	return a - s;
}

int main(int argc, char *argv[])
{
	long *ptr;

    assert(eclCompileSource(kernel) == eclSuccess);

	assert(eclMalloc((void **)&ptr, size * sizeof(long)) == eclSuccess);

	// Call the kernel
    size_t localSize = blockSize;
    size_t globalSize = size / blockSize;
    if(size % blockSize) globalSize++;
    globalSize *= localSize;
    cl_mem tmp = cl_mem(eclPtr(ptr));
    long val = 1;

    eclMemset(ptr, 0, size * sizeof(long));

    ecl_kernel kernel;

    assert(eclGetKernel("reset", &kernel) == eclSuccess);

    assert(eclSetKernelArg(kernel, 0, sizeof(mem), &tmp) == eclSuccess);
    assert(eclSetKernelArg(kernel, 1, sizeof(size), &size) == eclSuccess);
    assert(eclSetKernelArg(kernel, 2, sizeof(val), &val) == eclSuccess);
    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == eclSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	eclMemset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	fprintf(stderr,"Test full memset: ");
    memset(ptr, 0, size * sizeof(long));

    assert(eclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == eclSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	memset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

    eclReleaseKernel(kernel);

	eclFree(ptr);

    return 0;
}
