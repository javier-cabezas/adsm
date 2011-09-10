#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/cuda.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 2;
const size_t vecSizeDefault = 16 * 1024 * 1024;

unsigned nIter = 0;
size_t vecSize = 0;
const size_t blockSize = 512;

__global__ void vecAdd(float *c, const float *a, const float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}

struct thread_info {
    thread_t tid;
    const float *a, *b;
    float *c;
    size_t vecSize;
};

void *addVector(void *ptr)
{
	const float *a, *b;
    float *c;
    thread_info &info = *((thread_info *)ptr);
    a = info.a;
    b = info.b;
    c = info.c;
	size_t myVecSize = info.vecSize;

    printf("%p -> %p\n", a, gmacPtr(a));
    printf("%p -> %p\n", b, gmacPtr(b));
    printf("%p -> %p\n", c, gmacPtr(c));

	gmactime_t s, t;
	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg((unsigned int)myVecSize / blockSize);
	if(myVecSize % blockSize) Dg.x++;
	getTime(&s);
	vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), myVecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	return NULL;
}

int main(int argc, char *argv[])
{
	gmactime_t s, t;
    float *a, *b, *c;

	thread_info *nThread;
	unsigned n = 0;
	gmactime_t st, en;

    gmacError_t ret;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

    getTime(&s);
	// Alloc & init input data
	ret = gmacMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	// Alloc output data
	ret = gmacMalloc((void **)&c, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

    // Init the input data
    getTime(&s);
	valueInit(a, 1.0, vecSize);
	valueInit(b, 1.0, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

	nThread = (thread_info *)malloc(nIter * sizeof(thread_info));

	getTime(&st);
	for(n = 0; n < nIter; n++) {
        thread_t tid = thread_create(addVector, &nThread[n]);
		nThread[n].a = a + (vecSize / nIter) * n;
		nThread[n].b = b + (vecSize / nIter) * n;
		nThread[n].c = c + (vecSize / nIter) * n;
		nThread[n].tid = tid;
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n].tid);
	}

    getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");

    getTime(&s);
	gmacFree(a);
	gmacFree(b);
	gmacFree(c);
    getTime(&t);
    printTime(&s, &t, "Free: ", "\n");

    assert(error == 0.f);


	getTime(&en);
	printTime(&st, &en, "Total: ", "\n");

	free(nThread);

    return 0;
}
