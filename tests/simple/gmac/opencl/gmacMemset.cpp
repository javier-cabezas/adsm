#include <stdio.h>
#include <cstring>
#include <gmac/opencl.h>

const size_t size = 4 * 1024 * 1024;
const size_t blockSize = 32;

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

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	assert(oclMalloc((void **)&ptr, size * sizeof(long)) == gmacSuccess);

	// Call the kernel
    size_t localSize = blockSize;
    size_t globalSize = size / blockSize;
    if(size % blockSize) globalSize++;
    globalSize *= localSize;
    cl_mem tmp = cl_mem(oclPtr(ptr));
    long val = 1;

	fprintf(stderr,"GMAC_MEMSET\n");
	fprintf(stderr,"===========\n");
	fprintf(stderr,"Test full memset: ");
    oclMemset(ptr, 0, size * sizeof(long));

    OclKernel kernel;

    assert(__oclKernelGet("reset", &kernel) == gmacSuccess);

    assert(__oclKernelConfigure(&kernel, 1, NULL, &globalSize, &localSize) == gmacSuccess);
    assert(__oclKernelSetArg(&kernel, &tmp, sizeof(cl_mem), 0) == gmacSuccess);
    assert(__oclKernelSetArg(&kernel, &size, sizeof(size), 1) == gmacSuccess);
    assert(__oclKernelSetArg(&kernel, &val, sizeof(val), 2) == gmacSuccess);
    assert(__oclKernelLaunch(&kernel) == gmacSuccess);
    assert(oclThreadSynchronize() == gmacSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	oclMemset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	fprintf(stderr,"\n");
	fprintf(stderr,"LIBC MEMSET\n");
	fprintf(stderr,"===========\n");
	fprintf(stderr,"Test full memset: ");
    memset(ptr, 0, size * sizeof(long));

    assert(__oclKernelLaunch(&kernel) == gmacSuccess);
    assert(oclThreadSynchronize() == gmacSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	memset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	oclFree(ptr);

    return 0;
}
