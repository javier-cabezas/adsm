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

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	assert(gmacMalloc((void **)&ptr, size * sizeof(long)) == gmacSuccess);

	// Call the kernel
    size_t localSize = blockSize;
    size_t globalSize = size / blockSize;
    if(size % blockSize) globalSize++;
    globalSize *= localSize;
    cl_mem tmp = cl_mem(gmacPtr(ptr));
    long val = 1;

	fprintf(stderr,"GMAC_MEMSET\n");
	fprintf(stderr,"===========\n");
	fprintf(stderr,"Test full memset: ");
    gmacMemset(ptr, 0, size * sizeof(long));

    assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
    __oclPushArgument(&tmp, sizeof(cl_mem));
    __oclPushArgument(&size, sizeof(size));
    __oclPushArgument(&val, sizeof(val));
    assert(__oclLaunch("reset") == gmacSuccess);
    assert(gmacThreadSynchronize() == gmacSuccess);

	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	gmacMemset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	fprintf(stderr,"\n");
	fprintf(stderr,"LIBC MEMSET\n");
	fprintf(stderr,"===========\n");
	fprintf(stderr,"Test full memset: ");
    memset(ptr, 0, size * sizeof(long));

    assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
    __oclPushArgument(&tmp, sizeof(cl_mem));
    __oclPushArgument(&size, sizeof(size));
    __oclPushArgument(&val, sizeof(val));
    assert(__oclLaunch("reset") == gmacSuccess);
    assert(gmacThreadSynchronize() == gmacSuccess);

    gmacThreadSynchronize();
	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	memset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	gmacFree(ptr);

    return 0;
}
