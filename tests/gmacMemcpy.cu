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

void init(long *ptr, int s, long v)
{
	for(int i = 0; i < s; i++) {
		ptr[i] = v;
	}
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
	long *host = (long *)malloc(size * sizeof(long));
	assert(host != NULL);
	init(host, size, 1);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(size / blockSize);
	if(size % blockSize) Db.x++;

	fprintf(stderr,"Test full memcpy: ");
	memcpy(ptr, host, size * sizeof(long));
	reset<<<Dg, Db>>>(ptr, 1);
	fprintf(stderr, "%d\n", check(ptr, 2 * size));

	fprintf(stderr,"Test partial memcpy: ");
	memcpy(&ptr[size / 8], host, 3 * size / 4 * sizeof(long));
	fprintf(stderr, "%d\n", check(ptr, 5 * size / 4));

	fprintf(stderr,"Test reverse full: ");
	memcpy(host, ptr, size * sizeof(long));
	fprintf(stderr, "%d\n", check(host, 5 * size / 4));

	gmacFree(ptr);
	free(host);

    return 0;
}
