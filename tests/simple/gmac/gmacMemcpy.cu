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
	for(size_t i = 0; i < size; i++)
		a += ptr[i];
	return a - s;
}

int doTest(long *host, long *device, void *(*memcpy_fn)(void *, const void *, size_t n))
{
    init(host, size, 1);
    int ret1, ret2, ret3;

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(size / blockSize);
	if(size % blockSize) Db.x++;

	printf("Test full memcpy: ");
	memcpy_fn(device, host, size * sizeof(long));
	reset<<<Dg, Db>>>(gmacPtr(device), 1);
    gmacThreadSynchronize();
    ret1 = check(device, 2 * size);
	printf("%d\n", ret1);

	printf("Test partial memcpy: ");
	memcpy_fn(&device[size / 8], host, 3 * size / 4 * sizeof(long));
    ret2 = check(device, 5 * size / 4);
	printf("%d\n", ret2);

	fprintf(stderr,"Test reverse full: ");
	memcpy_fn(host, device, size * sizeof(long));
    ret3 = check(host, 5 * size / 4);
	fprintf(stderr, "%d\n", ret3);

    return (ret1 != 0 || ret2 != 0 || ret3 != 0);
}

static void *gmacMemcpyWrapper(void *dst, const void *src, size_t size)
{
	return gmacMemcpy(dst, src, size);
}

int main(int argc, char *argv[])
{
	long *ptr;
	long *host = (long *)malloc(size * sizeof(long));
	assert(host != NULL);

    // memcpy
	assert(gmacMalloc((void **)&ptr, size * sizeof(long)) == gmacSuccess);

    int res1 = doTest(host, ptr, memcpy);
    if (res1 != 0) fprintf(stderr, "Failed!\n");
	gmacFree(ptr);

    // gmacMemcpy
	assert(gmacMalloc((void **)&ptr, size * sizeof(long)) == gmacSuccess);
    int res2 = doTest(host, ptr, gmacMemcpyWrapper);
    if (res2 != 0) fprintf(stderr, "Failed!\n");
	gmacFree(ptr);

	free(host);

    return (res1 != 0 || res2 != 0);
}
