#include <stdio.h>
#include <gmac/cuda.h>

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
    int ret_full, ret_partial, ret_reverse;

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(size / blockSize);
	if(size % blockSize) Db.x++;

	fprintf(stderr, "Test full memcpy: ");
	memcpy_fn(device, host, size * sizeof(long));
	reset<<<Dg, Db>>>(gmacPtr(device), 1);
    gmacThreadSynchronize();
    ret_full = check(device, 2 * size);
	fprintf(stderr, "%d\n", ret_full);

	fprintf(stderr, "Test partial memcpy: ");
	memcpy_fn(&device[size / 8], host, 3 * size / 4 * sizeof(long));
    ret_partial = check(device, 5 * size / 4);
	fprintf(stderr, "%d\n", ret_partial);

	fprintf(stderr,"Test reverse full: ");
	memcpy_fn(host, device, size * sizeof(long));
    ret_reverse = check(host, 5 * size / 4);
	fprintf(stderr, "%d\n", ret_reverse);

    return (ret_full != 0 || ret_partial != 0 || ret_reverse != 0);
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

    int res_host = doTest(host, ptr, memcpy);
    if (res_host != 0) fprintf(stderr, "Failed!\n");
	gmacFree(ptr);

    // gmacMemcpy
	assert(gmacMalloc((void **)&ptr, size * sizeof(long)) == gmacSuccess);
    int res_device = doTest(host, ptr, gmacMemcpyWrapper);
    if (res_device != 0) fprintf(stderr, "Failed!\n");
	gmacFree(ptr);

	free(host);

    return (res_host != 0 || res_device != 0);
}
