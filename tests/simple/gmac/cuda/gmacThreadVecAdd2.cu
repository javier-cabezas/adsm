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

static float **s;

__global__ void vecAdd(float *c, const float *a, const float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}

struct thread_info {
    unsigned idx;
};

float *a, *b, *c;

void *addVector(void *ptr)
{
	thread_info *info = (thread_info*)ptr;
	gmactime_t s, t;

    float *local_a = &a[info->idx * (vecSize/nIter)];
    float *local_b = &b[info->idx * (vecSize/nIter)];
    float *local_c = &c[info->idx * (vecSize/nIter)];
	
    // Init the input data
    getTime(&s);
	valueInit(local_a, 1.0, vecSize);
	valueInit(local_b, 1.0, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg((unsigned int)vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;
	getTime(&s);
	vecAdd<<<Dg, Db>>>(gmacPtr(local_c), gmacPtr(local_a), gmacPtr(local_c), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");
 
	return NULL;
}

int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	gmactime_t st, en;

	gmacError_t ret = gmacSuccess;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

    getTime(&st);
	// Alloc & init input data
	ret = gmacMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	// Alloc output data
	ret = gmacMalloc((void **)&c, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	getTime(&en);
	printTime(&st, &en, "Alloc: ", "\n");

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	s = (float **)malloc(nIter * sizeof(float **));

	getTime(&st);
	for(n = 0; n < nIter; n++) {
		nThread[n] = thread_create(addVector, &s[n]);
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

    getTime(&st);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	getTime(&en);
	printTime(&st, &en, "Check: ", "\n");

    getTime(&st);
	gmacFree(a);
	gmacFree(b);
	gmacFree(c);
    getTime(&en);
    printTime(&st, &en, "Free: ", "\n");

    assert(error == 0.f);

	free(s);
	free(nThread);

    return 0;
}
