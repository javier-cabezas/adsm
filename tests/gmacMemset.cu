#include <stdio.h>
#include <gmac.h>

const size_t size = 4 * 1024 * 1024;
const size_t blockSize = 512;

__global__ void reset(long *a, long v)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= size) return;
	a[i] += v;
}

int check(long *ptr, int s)
{
	int a = 0;
	for(int i = 0; i < size; i++)
		a += ptr[i];
	return a - s;
}

int main(int argc, char *argv[])
{
	long *ptr;
	assert(gmacMalloc((void **)&ptr, size * sizeof(long)) == gmacSuccess);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(size / blockSize);
	if(size % blockSize) Db.x++;

	fprintf(stderr,"Test full memset: ");
	gmacMemset(ptr, 0, size * sizeof(long));
	reset<<<Dg, Db>>>(gmacPtr(ptr), 1);
	fprintf(stderr,"%d\n", check(ptr, size));

	fprintf(stderr, "Test partial memset: ");
	gmacMemset(&ptr[size / 8], 0, 3 * size / 4 * sizeof(long));
	fprintf(stderr,"%d\n", check(ptr, size / 4));

	gmacFree(ptr);

    return 0;
}
